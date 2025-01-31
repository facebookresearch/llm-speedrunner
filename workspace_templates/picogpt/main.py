#!/usr/bin/env python

# A GPT from scratch.
#
# This implementation aims at facilitating the test of new ideas. It
# should remain simple and self-explanatory even at the cost of
# performance.
#
# ``Premature optimization is the root of all evil.'' -- Donald Knuth

import os, sys, math, time, warnings

import argparse, tqdm

from dataclasses import dataclass

import torch

import torch.nn.attention.flex_attention as flex_attention

from torch import nn
from torch.nn import functional as F

from tasks import TaskArithmeticQuizz

######################################################################
# Model definition
######################################################################


class DummyTokenizer:
    def __init__(self, symbols: str | list | tuple) -> None:
        self.char2token = dict([(c, n) for n, c in enumerate(symbols)])
        self.token2char = dict([(n, c) for n, c in enumerate(symbols)])
        self.voc_size = len(symbols)

    def tsr2str(self, t: torch.Tensor) -> str | list[str]:
        if t.dim() == 2:
            return [self.tsr2str(x) for x in t]
        return "".join([self.token2char[x.item()] for x in t])

    def str2tsr(self, x: str | list[str]) -> torch.Tensor:
        if type(x) is list:
            t = [self.str2tsr(s) for s in x]
            l = max([x.size(0) for x in t])
            return torch.cat([F.pad(x, (0, l - x.size(0)))[None, :] for x in t])
        return torch.tensor([self.char2token[c] for c in x], dtype=torch.int64)


######################################################################


@dataclass
class SimpleGPTCache:
    # Length of the full sequence
    seq_len: int
    # Where we are in the full sequence
    t0: int
    # The stuff we stored for the different modules
    core: dict

    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len
        self.core = {}


######################################################################


class VaswaniPositionalEncoding(nn.Module):
    def __init__(self, len_max: int) -> None:
        super().__init__()
        self.len_max = len_max

    def forward(self, input: torch.Tensor, cache: dict) -> torch.Tensor:

        subseq_len = input.size(1)

        if cache is None:
            seq_len, t0, c = subseq_len, 0, None
        else:
            seq_len, t0, c = cache.seq_len, cache.t0, cache.core.get(self)

        if c is None:
            assert t0 == 0
            u = torch.arange(seq_len, device=input.device)[:, None]
            j = torch.arange(input.size(2), device=input.device)[None, :]
            k = j % 2
            s = u / (self.len_max ** ((j - k) / input.size(2))) + math.pi / 2 * k
            c = torch.sin(s)

            if cache is not None:
                cache.core[self] = c

        return input + c[None, t0 : t0 + subseq_len, :]


######################################################################


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_qk: int,
        dim_v: int,
        nb_heads: int,
        causal: bool,
        dropout: float,
    ) -> None:
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.dropout = dropout

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

        # We cache the mask for flex attention
        self.mask_config = None

    def forward(self, input: torch.Tensor, cache: SimpleGPTCache) -> torch.Tensor:
        q = torch.einsum("ntc,hdc->nhtd", input, self.w_q)
        k = torch.einsum("ntc,hdc->nhtd", input, self.w_k)
        v = torch.einsum("ntc,hdc->nhtd", input, self.w_v)

        N, H, seq_len, _ = q.size()

        subseq_len = seq_len

        if cache is None:
            t0 = 0
        else:
            seq_len, t0 = cache.seq_len, cache.t0

            c = cache.core.get(self)

            if c is None:
                assert t0 == 0
                k_cache = input.new(N, H, seq_len, k.size(3))
                v_cache = input.new(N, H, seq_len, v.size(3))
                cache.core[self] = (k_cache, v_cache)
            else:
                k_cache, v_cache = c

            k_cache[:, :, t0 : t0 + subseq_len, :] = k
            v_cache[:, :, t0 : t0 + subseq_len, :] = v

            k = k_cache[:, :, : t0 + subseq_len, :]
            v = v_cache[:, :, : t0 + subseq_len, :]

        # We cache the mask for flex attention
        if not self.mask_config == (seq_len, t0):

            def causal(b, h, q_idx, kv_idx):
                return q_idx + t0 >= kv_idx

            self.mask_config = (seq_len, t0)
            self.mask = flex_attention.create_block_mask(
                causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=t0 + 1
            )

        y = flex_attention.flex_attention(q, k, v, block_mask=self.mask)
        y = y.permute(0, 2, 1, 3).flatten(2)
        y = y @ self.w_o

        return y


######################################################################


class FeedForward(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc3 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc1(x)
        x3 = self.fc3(x)
        return self.fc2(F.silu(x1) * x3)


######################################################################


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim_model: int,
        dim_keys: int,
        dim_hidden: int,
        nb_heads: int,
        causal: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mha_norm = nn.RMSNorm((dim_model,))
        self.mha = MultiHeadAttention(
            dim_in=dim_model,
            dim_qk=dim_keys,
            dim_v=dim_model // nb_heads,
            nb_heads=nb_heads,
            causal=causal,
            dropout=dropout,
        )
        self.ffn_norm = nn.RMSNorm((dim_model,))
        self.ffn = FeedForward(
            dim_in=dim_model, dim_out=dim_model, dim_hidden=dim_hidden
        )

    def forward(self, input: torch.Tensor, cache: SimpleGPTCache) -> torch.Tensor:
        r = input

        x = self.mha_norm(r)
        x = self.mha(x, cache)
        r = r + x

        x = self.ffn_norm(r)
        x = self.ffn(x)
        r = r + x

        return r


######################################################################


@dataclass
class SimpleGPTArgs:
    voc_size: int
    dim_model: int = 512
    dim_keys: int = 128
    dim_hidden: int = 512
    nb_heads: int = 4
    nb_blocks: int = 4
    causal: bool = True
    dropout: float = 0.0
    len_max: int = 1e5


class SimpleGPT(nn.Module):
    def __init__(self, args: SimpleGPTArgs) -> None:
        super().__init__()
        self.voc_size = args.voc_size
        self.dim_model = args.dim_model
        self.dim_keys = args.dim_keys
        self.dim_hidden = args.dim_hidden
        self.nb_heads = args.nb_heads
        self.nb_blocks = args.nb_blocks
        self.causal = args.causal
        self.dropout = args.dropout
        self.len_max = args.len_max

        self.embedding = nn.Embedding(self.voc_size, self.dim_model)
        self.dropout = nn.Dropout(self.dropout)
        self.positional_encoding = VaswaniPositionalEncoding(self.len_max)

        self.trunk = nn.ModuleList(
            [
                TransformerBlock(
                    args.dim_model,
                    args.dim_keys,
                    args.dim_hidden,
                    args.nb_heads,
                    args.causal,
                    args.dropout,
                )
                for _ in range(args.nb_blocks)
            ]
        )

        self.readout = nn.Linear(in_features=args.dim_model, out_features=args.voc_size)

        with torch.no_grad():
            self.embedding.weight.normal_(mean=0, std=2e-2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = self.dropout(x)
        x = self.positional_encoding(x, cache=None)
        for tb in self.trunk:
            x = tb(x, cache=None)
        logits = self.readout(x)
        return logits

    def autoregression(
        self, tokens: torch.Tensor, ar_mask: torch.Tensor
    ) -> torch.Tensor:
        result = F.pad(tokens, (1, -1))
        cache = SimpleGPTCache(seq_len=result.size(1))

        for t0 in range(result.size(1)):
            cache.t0 = t0
            x = torch.zeros_like(result[:, 0:1]) if t0 == 0 else result[:, t0 - 1 : t0]
            x = self.embedding(x)
            x = self.dropout(x)
            x = self.positional_encoding(x, cache)

            for tb in self.trunk:
                x = tb(x, cache)

            logits = self.readout(x)
            dist = torch.distributions.categorical.Categorical(logits=logits)

            result[:, t0 : t0 + 1] = torch.where(
                ar_mask[:, t0 : t0 + 1], dist.sample(), tokens[:, t0 : t0 + 1]
            )

        return result


######################################################################
# Main train logic
#
# Run as python picogpt.py (prefer modifying default arguments, setting them to optimal values)
#
######################################################################


def main():
    parser = argparse.ArgumentParser(
        description="An implementation of a simple GPT with cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--seed", type=int, default=0)

    ########################################

    parser.add_argument("--nb_epochs", type=int, default=25)

    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--nb_train_samples", type=int, default=100000)

    parser.add_argument("--nb_test_samples", type=int, default=1000)

    # parser.add_argument("--optim", type=str, default="adam")

    parser.add_argument("--lr", type=float, default=1e-4)

    ########################################

    parser.add_argument("--model", type=str, default="50M")

    parser.add_argument("--dim_model", type=int, default=None)

    parser.add_argument("--dim_keys", type=int, default=None)

    parser.add_argument("--dim_hidden", type=int, default=None)

    parser.add_argument("--nb_heads", type=int, default=None)

    parser.add_argument("--nb_blocks", type=int, default=None)

    parser.add_argument("--dropout", type=float, default=0.1)

    ######################################################################

    args = parser.parse_args()

    ######################################################################

    default_model_args = {
        "17K": {
            "dim_model": 32,
            "dim_keys": 32,
            "dim_hidden": 32,
            "nb_heads": 2,
            "nb_blocks": 2,
        },
        "6M": {
            "dim_model": 256,
            "dim_keys": 64,
            "dim_hidden": 1024,
            "nb_heads": 4,
            "nb_blocks": 6,
        },
        "50M": {
            "dim_model": 512,
            "dim_keys": 64,
            "dim_hidden": 2048,
            "nb_heads": 8,
            "nb_blocks": 12,
        },
        "250M": {
            "dim_model": 1024,
            "dim_keys": 128,
            "dim_hidden": 2048,
            "nb_heads": 8,
            "nb_blocks": 24,
        },
        "500M": {
            "dim_model": 1024,
            "dim_keys": 128,
            "dim_hidden": 2048,
            "nb_heads": 8,
            "nb_blocks": 48,
        },
    }

    if args.model in default_model_args:
        for k, v in default_model_args[args.model].items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    else:
        raise ValueError(f"Unknown model {args.model}")

    ######################################################################

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    ######################################################################

    def log_string(s):
        t = time.strftime("%Y%m%d-%H%M%S ", time.localtime())

        print(t + s)
        sys.stdout.flush()


    log_string(f"argv {' '.join(sys.argv)}")

    for n in vars(args):
        log_string(f"args.{n} {getattr(args, n)}")


    ######################################################################

    optimizer = torch.optim.Adam

    ######################################################################
    # Init stuff

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ######################################################################
    # create data, nothing after the prompt for the test

    task = TaskArithmeticQuizz(nb_numbers=4, valmax=12)

    train_data = [task.generate_sample() for _ in range(args.nb_train_samples)]

    for s in train_data:
        assert task.correct(s)

    test_data = [task.generate_sample() for _ in range(args.nb_test_samples)]

    ######################################################################
    # create tokenizer, model and data loaders

    tokenizer = DummyTokenizer(task.used_characters())

    model_args = SimpleGPTArgs(
        voc_size=tokenizer.voc_size,
        dim_model=args.dim_model,
        dim_keys=args.dim_keys,
        dim_hidden=args.dim_hidden,
        nb_heads=args.nb_heads,
        nb_blocks=args.nb_blocks,
        dropout=args.dropout,
    )

    model = SimpleGPT(model_args)
    optim = optimizer(model.parameters(), lr=args.lr)

    train_data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tokenizer.str2tsr(train_data)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tokenizer.str2tsr(test_data)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.to(device)

    ######################################################################
    # train

    nb_parameters = sum([p.numel() for p in model.parameters()])
    log_string(f"nb_parameters {nb_parameters}")
    log_string(f"device {device}")

    # Prints a few training samples
    train_input = next(iter(train_data_loader))[0][:10].to(device)
    for x in train_input:
        log_string(f"train_sample " + tokenizer.tsr2str(x))

    for n_epoch in range(args.nb_epochs):

        acc_train_samples, acc_train_loss = 0, 0.0
        for input in train_data_loader:
            input = input[0].to(device)
            logits = model(F.pad(input, (1, -1)))
            loss = F.cross_entropy(logits.transpose(1, 2), input)
            acc_train_loss += loss.item() * input.size(0)
            acc_train_samples += input.size(0)
            optim.zero_grad()
            loss.backward()
            optim.step()

        acc_test_samples, acc_test_loss = 0, 0.0
        for input in test_data_loader:
            input = input[0].to(device)
            logits = model(F.pad(input, (1, -1)))
            loss = F.cross_entropy(logits.transpose(1, 2), input)
            acc_test_loss += loss.item() * input.size(0)
            acc_test_samples += input.size(0)

        ######################################################################
        # generate a bunch

        train_loss = acc_train_loss/acc_train_samples
        test_loss = acc_test_loss/acc_test_samples

        nb_test_for_accuracy = 1000

        acc_test_samples, acc_test_correct = 0, 0
        for input in test_data_loader:
            input = input[0].to(device)
            c = (input == tokenizer.char2token[":"]).long()
            ar_mask = (c.cumsum(dim=1) - c) > 0
            generated = model.autoregression(input, ar_mask)

            for x in tokenizer.tsr2str(generated):
                if task.correct(x):
                    acc_test_correct += 1
                    comment = " [valid]"
                else:
                    comment = ""

                if acc_test_samples < 10:
                    log_string(f"generated " + x + comment)
                acc_test_samples += 1

            if acc_test_samples >= nb_test_for_accuracy:
                break

        test_acc = acc_test_correct/acc_test_samples
        log_string(
            f"epoch:{n_epoch} train loss:{train_loss} test loss:{test_loss} test acc:{test_acc}"
        )


if __name__ == '__main__':
    main()

