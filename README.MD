# pleonasty

A very, very, very, very basic library to abstract interactions with an LLM for single-response purposes. Like, if you're a computer scientist, this package will probably make things harder rather than easier. But, if you're like me, then this is great!

In essence, this is a library that makes it a bit easier to load up a "chat" or "instruct" LLM and then have it sequentially provide a single response to multiple input texts. For example, if you want to use an LLM to "code" or annotate texts in the same way that a human would, you might want to give it the same instructions before batch coding an entire dataset. This makes it relatively easy to do so, saving the output as a CSV file.

## Installation

The easiest way to get up and running with `pleonasty` is to install via `pip`, e.g.,

```pip install pleonasty```

Note that, in order to use this package, you will already need to have your CUDA environment properly configured if you plan to use a GPU for accelerated inference. This includes having the appropriate version of PyTorch installed with CUDA support.

To use pleonasty, ensure you have the following installed:

- Python 3.10 or higher (might work with older versions, but not tested)
- PyTorch with CUDA support (if using a GPU)

All other requirements can be found in the `pyproject.toml` file.

## How to Use

An example notebook is included in this repo that shows how it can be used. I have also included a "chat mode" where you can load up an LLM and have back-and-forth interactions with it — an example of this is also provided in a sample notebook.
