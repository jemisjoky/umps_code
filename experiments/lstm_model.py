import sys

import jax
import torch
import numpy as np
import torch.nn as nn
import jax.numpy as jnp

sys.path.append('..')
from train_tools import to_string

def jax2tor(array, tor2jax=False):
    """Convert between JAX arrays and Pytorch tensors"""
    if tor2jax:
        return jnp.array(array.numpy())
    else:
        return torch.tensor(np.array(array))

def strset2tor(strset, in_dim):
    """
    Convert a StrSet object to Pytorch onehot tensor and str_lens vector
    """
    index_mat = jax2tor(strset.index_mat).T.long()
    str_lens  = jax2tor(strset.str_lens).long()
    index_tens = torch.zeros(index_mat.shape + (in_dim,))
    index_tens = index_tens.scatter(2, index_mat[:, :, None], 1)

    return index_tens, str_lens

class ProbLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bi_dir=False, 
                 pos_enc=False, pe_dim=6, **kwargs):
        """
        LSTM model trained to predict characters given surrounding context
        
        Note that the model in question can take in StrSet data, to be 
        compatible with the framework I've already built up

        Some parts based on github.com/salesforce/awd-lstm-lm
        """
        super().__init__()

        # Define a base LSTM, inital hidden and cell states, and a decoder
        self.bi_dir = bi_dir
        self.num_dir = 2 if bi_dir else 1
        assert 'dropout' in kwargs
        assert pe_dim % 2 == 0      # Only dealing with even-dim encodings
        self.iesize = input_size + (pe_dim if pos_enc else 0)
        self.lstm = torch.nn.LSTM(input_size=self.iesize, 
                                  hidden_size=hidden_size, 
                                  num_layers=num_layers, 
                                  bidirectional=bi_dir, **kwargs)
        self.init_hidden = nn.Parameter(torch.empty(num_layers*self.num_dir, 1, hidden_size))
        self.init_cell = nn.Parameter(torch.empty(num_layers*self.num_dir, 1, hidden_size))

        # Whether or not to use a concatenated positional encoding
        assert not (bi_dir and pos_enc)
        self.pos_enc = pos_enc
        self.pe_dim = pe_dim

        # Define a decoder for the model
        self.decoder = nn.Sequential(nn.Linear(hidden_size*self.num_dir, input_size),
                                     nn.Softmax(dim=2))

        # Initialize our initial hidden and cell states randomly
        initrange = 0.1
        self.init_hidden.data.uniform_(-initrange, initrange)
        self.init_cell.data.uniform_(-initrange, initrange)

        # Hyperparameters to remember
        self.hsize = hidden_size
        self.isize = input_size
        
    def forward(self, inp_strset, log_format=True):
        """
        Obtain log probabilities for each character in StrSet of strings

        Note that this assumes the inputs are prepended/appended with
        BOS and EOS tokens, and the returned probabilities are only for
        the chars after the initial BOS. The EOS probability isn't included
        for the bidirectional model, but is for the unidirectional model
        """
        # Convert input from StrSet to onehot data tensor and str_lens vec
        in_dim = self.isize
        inp_data, str_lens = strset2tor(inp_strset, in_dim)
        max_len, batch_size, inp_data_dim = inp_data.shape

        # Concatenate positional encoding to input strings
        if self.pos_enc:
            inp_data = self.concat_posenc(inp_data, str_lens)

        # Evaluate LSTM on input sequence
        init_h = self.init_hidden.expand(-1, batch_size, -1)
        init_c = self.init_cell.expand(-1, batch_size, -1)
        seq_out, _ = self.lstm.forward(inp_data, (init_h, init_c))

        # Rearrange our output, depending on our directionality. This
        # ensures the output at each site doesn't directly contain 
        # information from the site itself
        if self.bi_dir:
            seq_out = seq_out.view(max_len, batch_size, 2, self.hsize)
            forw_seq, back_seq = seq_out[:, :, 0], seq_out[:, :, 1]
            seq_out = torch.stack((forw_seq[:-2], back_seq[2:]), dim=2)
            seq_out = seq_out.view(max_len-2, batch_size, 2*self.hsize)
        else:
            seq_out = seq_out[:-1]

        # Use our decoder to convert hidden states to probabilities
        all_probs = self.decoder(seq_out)

        if log_format:
            probs_obj, fill_val = torch.log(all_probs), 0.
        else:
            probs_obj, fill_val = all_probs, 1.

        # Fill all entries beyond the end of sequence with 0 or 1
        len_diff = 2 if self.bi_dir else 1
        str_lens, max_len = str_lens - len_diff, max_len - len_diff
        too_long = torch.arange(max_len)[:, None] >= str_lens[None, :]
        probs_obj = torch.where(too_long[..., None].expand(-1, -1, in_dim),
                                torch.full_like(probs_obj, fill_val), probs_obj)

        return probs_obj

    def get_loss(self, inp_strset, rescale_probs=False):
        """
        Evaluate sum of NLL loss for each character in inp_strset

        This assumes input StrSet begins and ends with BOS and EOS chars, 
        with the prob of the BOS char not included in output loss. The 
        prob of the EOS char is only included for the unidirectional model
        """
        # Get log probabilities over all characters
        str_lens = jax2tor(inp_strset.str_lens)
        indices = jax2tor(inp_strset.index_mat).T.long()
        max_len, batch_size = indices.shape
        all_probs = self.forward(inp_strset, log_format=False)

        # Remove initial chars (and maybe final chars) for evaluation, and
        # rescale probabilities if needed to disallow BOS/EOS chars
        if self.bi_dir:
            # str_lens = str_lens - 2
            indices = indices[1:-1]
            max_len = max_len - 2
            if rescale_probs:
                all_probs[:, :, -2:] = 0
                all_probs = all_probs / torch.sum(all_probs, dim=2, keepdim=True)
        else:
            str_lens = str_lens - 1
            indices = indices[1:]
            max_len = max_len - 1
            if rescale_probs:
                cond_vec = torch.arange(self.isize) < (self.isize - 2)
                new_probs = torch.where(cond_vec[None, None], all_probs, 
                                        torch.zeros(()))
                new_probs = new_probs / torch.sum(new_probs, dim=2, keepdim=True)
                cond_mat = (torch.arange(len(all_probs))[:, None] == 
                                                (str_lens-1)[None].long())
                all_probs = torch.where(cond_mat[..., None], all_probs, new_probs)

        # Get log probabilities of correct chars
        log_probs = torch.log(all_probs)
        seq, batch = np.mgrid[0:max_len, 0:batch_size]
        scores = log_probs[seq, batch, indices]

        # Sum log probs over characters in each sentence
        return -torch.sum(scores, dim=0)

    @torch.no_grad()
    def sample(self, rng_key, alphabet, samp_mode='variable', num_samps=1, 
                     samp_len=None, ref_strset=None):
        """
        Produce a list of strings from the LSTM model using sampling mode

        For a unidirectional model, available modes are 'variable' and 
        'fixed', which respectively sample until the EOS token, or sample
        a fixed length while forbidding sampling of the EOS token. The
        'fixed' mode requires samp_len to be set, and both require num_samps.

        For a bidirectional model, the only available mode is 'completion',
        which takes in StrSet corresponding to a list strings and returns 
        larger list of strings containing the model's completion of each
        string when each of its characters have been masked out.

        Args:
            rng_key:    JAX PRNGKey for randomization
            alphabet:   List of characters defining our sampling alphabet
            samp_mode:  String specifying particular means of sampling, 
                        either 'variable', 'fixed', or 'completion'
            num_samps:  Number of samples to generate, required for 
                        'variable' and 'fixed' modes
            samp_len:   Target length of samples, required for 'fixed', and
                        for 'variable' when positional encoding is used.
                        This length doesn't include BOS/EOS chars, which
                        aren't returned by the sampler
            ref_strset: StrSet encoding the strings we wish to fill in,
                        required for 'completion'. All encoded strings are
                        assumed to begin/end with a BOS/EOS character

        Returns:
            samples:    List of strings without BOS/EOS chars, which:
                        * For 'variable', list of variable-length strings.
                        * For 'fixed', list of fixed-length strings.
                        * For 'completion', long list of strings where 
                          each string in ref_strset is repeated many times
                          with each character sampled from model using 
                          surrounding characters as context
        """
        if self.bi_dir:
            # Check input, get strings with stripped 
            assert samp_mode == 'completion'
            assert ref_strset is not None
            # TODO: Deal with BOS/EOS issues
            ref_strs = [s[1:-1] for s in to_string(ref_strset, alphabet)]
            # ref_strs = to_string(ref_strset, alphabet)

            char_probs = self.forward(ref_strset, log_format=False)
            assert len(char_probs.shape) == 3
            # Condition on no BOS or EOS char
            char_probs[:, :, -2:] = 0
            char_probs = char_probs / torch.sum(char_probs, 2, keepdim=True)
            cum_probs = torch.cumsum(char_probs, axis=2)

            # TODO: Account for variable length ref_strset
            num_strs, str_len = ref_strset.index_mat.shape
            str_len = str_len - 2   # Don't make predictions for BOS or EOS
            
            # Sample chars to fill in each space of each sequence in ref_strset
            rand_floats = jax.random.uniform(rng_key, shape=(str_len, num_strs))
            # rand_floats = jax2tor(rand_floats)
            samp_ints = np.argmax(jax2tor(cum_probs, tor2jax=True) > rand_floats[..., None], axis=2).T
            samp_ints = jax2tor(samp_ints)
            samp_chars = [[alphabet[i] for i in seq] for seq in samp_ints]

            # TODO: Fix to ensure we return big list of filled-in strings
            samples = [s[:i] + c + s[i+1:] for s, cs in zip(ref_strs, samp_chars)
                        for i, c in enumerate(cs)]

            # Check that we're not making predictions for BOS and EOS
            assert all(len(s) == str_len for s in samples)

            return samples

        else:
            # Unpack args and state, build character lookup
            assert num_samps is not None
            assert samp_mode in ['variable', 'fixed']
            fixed = samp_mode == 'fixed'
            if fixed: assert samp_len >= 0
            char2ind = {c: i for i, c in enumerate(alphabet)}
            bos, eos = [alphabet[i] for i in (-2, -1)]
            h = self.init_hidden.expand(-1, num_samps, -1)
            c = self.init_cell.expand(-1, num_samps, -1)
            pos_enc = self.pos_enc
            if pos_enc: assert samp_len is not None
            hidden = (h, c)

            # Function which determines if we're finished sampling
            if fixed:
                stop_cond = lambda samps: all(len(s) == samp_len + 2 
                                              for s in samps)
            else:
                stop_cond = lambda samps: all(eos in s for s in samps)

            # Sample characters one by one until stopping condition is hit
            n = 0
            samples = ["^"] * num_samps
            while not stop_cond(samples):
                # Build onehot LSTM input from last chars of samples
                last_chars = torch.tensor([char2ind[s[-1]] for s in samples])
                char_vecs = torch.zeros(num_samps, self.isize)
                char_vecs.scatter_(1, last_chars[:, None], 1)
                char_vecs = char_vecs[None]  # LSTM needs length index

                # Add on positional encoding
                if pos_enc:
                    char_vecs = self.concat_posenc(char_vecs, samp_len, n)

                # Call LSTM on our single character, get char probs
                h_out, hidden = self.lstm.forward(char_vecs, hidden)
                char_probs = self.decoder(h_out)[0]

                # Condition on no BOS char
                char_probs[:, -2] = 0
                char_probs = char_probs / torch.sum(
                                            char_probs, 1, keepdim=True)

                # For fixed mode, either add a EOS or condition on no EOS
                if fixed:
                    if n == samp_len:
                        samples = [s + eos for s in samples]
                        continue
                    else:
                        char_probs[:, -1] = 0
                        char_probs = char_probs / torch.sum(char_probs, 
                                                        1, keepdim=True)
                        n += 1

                cum_probs = np.cumsum(char_probs.numpy(), axis=1)

                # Sample a new char for each sample string
                rng_key, key = jax.random.split(rng_key)
                rand_floats = jax.random.uniform(key, shape=(num_samps,))
                samp_ints = np.argmax(cum_probs > rand_floats[:, None], axis=1)
                samp_chars = [alphabet[i] for i in samp_ints]
                samples = [s + alphabet[i] for s, i in zip(samples, samp_ints)]

            # Remove initial BOS and trim to first EOS, check samples
            if fixed:
                assert all(s[0] == bos and s[-1] == eos for s in samples)
                samples = [s[1:-1] for s in samples]
                assert set(len(s) for s in samples) == {samp_len}
            else:
                assert all(s[0] == bos for s in samples)
                samples = [s.split(eos)[0][1:] for s in samples]
            assert all(bos not in s and eos not in s for s in samples)

            return samples

    def concat_posenc(self, inp_data, str_lens, start=0):
        """
        Concatenate data with positional encoding of desired dimension, 
        starting from the position `start`
        """
        pe_dim = self.pe_dim
        half_dim = pe_dim // 2
        assert 2 * half_dim == pe_dim
        max_len, batch_size, inp_dim = inp_data.shape
        if not isinstance(str_lens, torch.Tensor):
            str_lens = torch.tensor(str_lens).expand(batch_size)

        # Holds the position of each input
        counter = torch.arange(start, start+max_len).float()
        counter = counter[:, None, None].expand(-1, batch_size, half_dim)

        # Frequencies associated with each entry of positional encoding
        # Using similar encoding as the one in Attention is All You Need
        freqs = torch.arange(half_dim).float() * (-2 / pe_dim)
        str_lens = str_lens[None, :, None].float()
        freqs = torch.pow(str_lens, freqs[None, None])
        trig_args = (3.14159 / 2) * counter * freqs

        # The positional encoding itself
        enc_data = torch.zeros(max_len, batch_size, pe_dim)
        enc_data[..., 0::2] = torch.sin(trig_args)
        enc_data[..., 1::2] = torch.cos(trig_args)

        # Return input concatenated with positional encodings
        return torch.cat((inp_data, enc_data), dim=2)

    def eval(self):
        self.lstm = self.lstm.eval()
        return self

    def train(self):
        self.lstm = self.lstm.train()
        return self

# if __name__ == '__main__':
#     input_dim  = 1
#     batch_dim  = 1
#     bond_dim   = 2
#     pe_dim     = 6
#     max_len    = 10
#     my_lstm = ProbLSTM(input_dim, bond_dim, 1, 
#                        bi_dir=False, dropout=0, 
#                        pos_enc=True, pe_dim=pe_dim)

#     inp_data = torch.zeros(max_len, batch_dim, input_dim)
#     str_lens = torch.full((batch_dim,), max_len)

#     print(my_lstm.concat_posenc(inp_data, str_lens, start=0))