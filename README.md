# Python: A Simple Chat Robot Based on a Seq2Seq Model

Data source: https://huggingface.co/datasets/daily_dialog
Depend on [d2l](https://d2l.ai/).

## Process Dialog
In `process_dialogue.py`

Raw data shortcut:
```
Say , Jim , how about going for a few beers after dinner ? __eou__ You know that is tempting but is really not good for our fitness . __eou__ What do you mean ? It will help us to relax . __eou__ Do you really think so ? I don't . It will just make us fat and act silly . Remember last time ? __eou__ I guess you are right.But what shall we do ? I don't feel like sitting at home . __eou__ I suggest a walk over to the gym where we can play singsong and meet some of our friends . __eou__ That's a good idea . I hear Mary and Sally often go there to play pingpong.Perhaps we can make a foursome with them . __eou__ Sounds great to me ! If they are willing , we could ask them to go dancing with us.That is excellent exercise and fun , too . __eou__ Good.Let ' s go now . __eou__ All right . __eou__  
Can you do push-ups ? __eou__ Of course I can . It's a piece of cake ! Believe it or not , I can do 30 push-ups a minute . __eou__ Really ? I think that's impossible ! __eou__ You mean 30 push-ups ? __eou__ Yeah ! __eou__ It's easy . If you do exercise everyday , you can make it , too . __eou__
```
Processing goal:
```
say , jim , how about going for a few beers after dinner ?  you know that is tempting but is really not good for our fitness .  
what do you mean ? it will help us to relax .   do you really think so ? i don't . it will just make us fat and act silly . remember last time ?  
i guess you are right.but what shall we do ? i don't feel like sitting at home .    i suggest a walk over to the gym where we can play singsong and meet some of our friends .  
that's a good idea . i hear mary and sally often go there to play pingpong.perhaps we can make a foursome with them .   sounds great to me ! if they are willing , we could ask them to go dancing with us.that is excellent exercise and fun , too .  
good.let ' s go now .   all right .  
can you do push-ups ?   of course i can . it's a piece of cake ! believe it or not , i can do 30 push-ups a minute .  
really ? i think that's impossible !    you mean 30 push-ups ?  
yeah !  it's easy . if you do exercise everyday , you can make it , too .
```
## Build Dataloader
In `build_vocab.py` and `prepare_data.py`
1. Tokenize by spaces.
2. Use `d2l.Vocab` to build a vocabulary and save locally.
3. Encode and add BOS_TOKEN and EOS_TOKEN.
4. Build the dataset with `Dataset` from `torch.utils.data`.
5. Build the dataloader with `DataLoader` from `torch.utils.data`. 
	- Define `collate_fn` function to truncate and add padding tokens. The length of the sentences in the same batch is equal to `num_steps` defined in `parameters.py`.
	- Define `batch_sampler` function to create pool of indices with similar lengths.
	- `batch_size` is defined in `parameters.py`.
## Build Model
In `model.py`

The Seq2Seq model includes an encoder layer and a decoder layer. The ouput of encoder is the input of decoder.

## Train Model and Evaluate
In `train_eval.py`

`num_epochs` is defined in `parameters.py`.

Force teaching is on when the given `force_teaching_ratio` is bigger than the random number.

In each epoch, the dataloader is iterated.

Each batch enters the model and gets the output.

Train loss is calculated with `MaskedSoftmaxCELoss` defined in `model.py`, inheriting from `torch.nn.CrossEntropyLoss`.

Evaluating process is similar.
## Run
```
Input:  Christmas is coming. They must be popular again this season.
Output:  bat smile smile men smile take-out smile bat interviewed smile interviewed men
```