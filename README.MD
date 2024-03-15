# pleonasty

A very, very, very, very simple library to abstract interactions with an LLM for single-response purposes.
In essence, this is a library that makes it a bit easier to load up a "chat" or "instruct" LLM and then sequentially have it provide a response to multiple input texts. For example, if you want to use an LLM to "code" texts in the same way that a human would, you might want to give it the same instructions before batch coding an entire dataset. This makes it relatively easy to do so, saving the output as a CSV file.