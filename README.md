# Transpeare
This is a transformer model which we built from scratch. which is trained on shakespearian text. The output of this model will be very similar to shakespearian writting style but the output would gibberish :). Well, not totally gibberish but very similar to shakespearian text.

## Architecture of the model
This is a character level model. And contrary to the typical architecture of the model, this is a only decoder transformer model because we didn't needed aything to encode. We are just training our model on the shakespeare text which is about one million characters. This model can be trained on any text and it will output the text similar to the text style given to it.

<p align="center">
<img width="348" alt="Screenshot 2024-02-10 at 1 28 03 PM" src="https://github.com/davnish/transpeare/assets/32027279/c99823cf-3af5-486c-a459-a208292e1b0b">
</p>

## Model
The model starts with class `BigramLanguageModel` it then calls the class `Blocks` which is used to make multiple copies of the middle block as you cna see in the above diagram; with one `masked_multi_headed_attention` and one `feed_forward` and two `layer_norms`, The Nx in the diagram signifies that. each `Block` calls for the `MultiHeadedAttention`  and each multi_head class the `Head` class which calculates the attention for the single head. There is a `estimate_loss` class which calculates the `loss` and `val_loss` after every `eval_intervals`. 

## Output
This is an example output from the model.
```
Upon his softice and treasons from which cark,
Coming that let him good words and brood
As if he scarce this man yet did show his sea,
Might beat as liht as it basward to our feel,
Albow selfsame us; to look King Henry lief
With toger-faced prophece, or your lifty dog,
not fairly your could dreams soldiers wind hope.

JULIET:
O, look my grave you, my lord.

QUEEN:
My lordship is change to wail, and to prepared,
Divide mine, that reverend smiles his King Richard king.
```
yeah LOL :P

## Training
The final model was trained on Nvidia Quadro RTX 4000 and took about 30 mins. On the hyperparameters given in the `train.py`. If you are using apple chips you can use `mps` just set the device to `mps`. Use `torch.backends.mps.is_available` to verify. As as for `cuda` settings are already set. And if you just have a `cpu` just decrease the hyperparameter you will be good to go. The final loss I achieved was 1.499. If you wanna use the model I have saved the model as `modelv1.pt`. Just set the `load_model=True` in `train.py` and if it is set as `False` then model will go into training mode. I have used approx 10 million parameters for training, 10,788,161 to be exact.

## Libraries
- `pytorch`


## Resources
This model is based on tutorials of the legend himself Andrej karpathy, on his makemore series, check it out!
https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&ab_channel=AndrejKarpathy
https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&ab_channel=AndrejKarpathy
