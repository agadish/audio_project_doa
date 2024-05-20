Advanced Topics in Audio Processing using Deep Learning
Final Project - HOWTO
* Students: Assaf Gadish      Tomer Lerman      Dor Siman Tov
* Github URL: https://github.com/agadish/audio_project_doa/
* Implemented Paper: Multi-Microphone Speaker Separation based on Deep DOA Estimation (Chazan et al.)

-----------------------------------------
| Hardware requirements
-----------------------------------------
24GB RAM, 30GB storage
GPU with dedicated 14GB RAM may significantly help (for --batch-size=64)


-----------------------------------------
| Data generation
-----------------------------------------
1. In the directory "source_signal", download and extract the following datasets (expected structure: source_signal/LibriSpeech/train-clean-100, ...)
  a) LibriSpeech train-clean:
     https://www.openslr.org/resources/12/train-clean-100.tar.gz
  b) LibriSpeech test-other:
     https://www.openslr.org/resources/12/test-other.tar.gz
2. In the directory "timit_rir", download and extract the following rir responses:
  a) Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3.zip:
    https://drive.google.com/file/d/1THNdtrzy9WCIUZfjj5aIWXUYg8g9Eesy/view?usp=drive_link
  b) Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.360s)_3-3-3-8-3-3-3.zip:
     https://drive.google.com/file/d/1Bc97XSIebnsdCnQQzctqRcEmh74uBUcE/view?usp=drive_link

3. Within the project main directory extract within your python 3.10 (that has requirements.txt installed):
   > python audio_proj_generator.py
   
   The program has default arguments which should work out of the box, their documentation can be seen by running:
   > python audio_proj_generator.py --help


-----------------------------------------
| Train & Test                          |
-----------------------------------------

Basic usage
-----------
python main.py --train --test --device=cuda

We use a single script for both train and test - its called main.py.
It supports train only (--train), test only (--test), or train and test (--train --test) where we test the model we just trained.

You may pass "--checkpoint-path" to start the training from it, or may leave it empty for the model to train from raw weights.
On train+test mode, the model after the train will be used for test.
On test-only mode you must pass checkpoint-path.


Test metrics
------------
We use SDR and SIR.
For the test script, the program will print a table of values for both extracted speaker signals.
The first index in the matrice is SDR=0 and SIR=1, and the seconnd is the speaker_a=0 and speaker_b=1.
The data is divided by:
a. mix=mixture of signals (noisy signals), sep=separated signals (separated speakers)
b. speakers radius (1[m] or 2[m]) 
c. reverb value (0.16[s] or 0.36[s]).

For example: sep_rad1.0_rev0.16_1_1_epoch refers to separated signals with radius=1[m] reverb=0.16[s], SIR for speaker B.

Tensorboard flow
----------------
Run in a termianl from the project directory (with the python environment):
> tensorboard --logdir=tb_logs
See the results at http://localhost:6006/

Full parameters documentation
-----------------------------
All the supported params can be seen by runnning:
> python main.py --help
usage: main.py [-h] [--train] [--batch-size BATCH_SIZE] [--lr LR] [--max-epochs MAX_EPOCHS] [--early-stopping-patience EARLY_STOPPING_PATIENCE] [--train-pt-prefix TRAIN_PT_PREFIX]
               [--val-pt-prefix VAL_PT_PREFIX] [--test] [--test-pt-prefix TEST_PT_PREFIX] [--checkpoint-path CHECKPOINT_PATH] [--data-dir DATA_DIR] [--device DEVICE]

DOA model trainer/tester

options:
  -h, --help            show this help message and exit
  --train               Enable training mode
  --batch-size BATCH_SIZE
                        Batch size for training (default: 64)
  --lr LR               Learning rate for training (default: 1e-3)
  --max-epochs MAX_EPOCHS
                        Max epochs for training (default: 100)
  --early-stopping-patience EARLY_STOPPING_PATIENCE
                        Early stopping patience for training (default: 3)
  --train-pt-prefix TRAIN_PT_PREFIX
                        Prefix of train data "[prefix]_[index].pt" files on data dir (default: "train06r076v3")
  --val-pt-prefix VAL_PT_PREFIX
                        Prefix of validation data "[prefix]_[index].pt" files on data dir (default: "validation06r076v3")
  --test                Enable testing mode
  --test-pt-prefix TEST_PT_PREFIX
                        Prefix of test data "[prefix]_[index].pt" files on data dir (default: "test2r0168v4")
  --checkpoint-path CHECKPOINT_PATH
                        Path to the model checkpoint for testing
  --data-dir DATA_DIR   Directory of data batches (default: "data_batches")
  --device DEVICE       Device to load/run the model, should be "cpu" or "cuda" (default: "cuda")
