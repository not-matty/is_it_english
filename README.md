
## Task: train a model to classify which of two English chunks of text have been altered. 

Current implementation: decoder-only, byte-level LM -> sum nll and choose lower loss as original. Trained over only originals with causal mask. ~62M paramters achieves 81.6 percent accuracy on 900k samples. Basic nanoGPT fork. 

#Todo: 
- Try appending original and altered together and train without causal mask, with logistic loss (essentially cross attention), or just try encoder style LM without causal mask.
- Train logistic regression on top of pretrained LM.
- Train a tokenizer, and try on bigger vocab size. 