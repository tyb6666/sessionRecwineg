### Installing
Clone the repository and run `pip install -r requirements.txt`. 

No manual dataset download is required.

### Executing
RQ_VAE tokenizer model and the retrieval model are trained separately, using two separate training scripts. 
#### Sample usage
To train both models on the **Amazon Reviews** dataset, run the following commands:
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py configs/rqvae_amazon.gin`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py configs/decoder_amazon.gin`

To train both models on the **MovieLens 32M** dataset, run the following commands:
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py configs/rqvae_ml32m.gin`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py configs/decoder_ml32m.gin`


