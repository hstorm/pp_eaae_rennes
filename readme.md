

## Code EAAE 2023 Keynote **Thomas Heckelei** [Program Link](https://eaae2023.colloque.inrae.fr/keynote-speakers/thomas-heckelei)
# *Probabilistic programming for embedding theory and quantifying uncertainty in statistical models*, 

Hugo Storm, **Thomas Heckelei**, Kathy Baylis

Questions: *hugo.storm@ilr.uni-bonn.de*



## Example code

*To run the examples, check the notes below on how to setup a python environment.*

1. Common econometrics model implemented in numpyro [econometrics_in_pp.py](https://github.com/hstorm/pp_eaae_rennes/blob/master/examples/econometrics_in_pp.py)

2. Cumulative prospect theory model implemented in numpyro [cumulative_prospect_theory.py](https://github.com/hstorm/pp_eaae_rennes/blob/master/examples/cumulative_prospect_theory.py)

3. Prior sampling in numpyro  using regional yield  data [prior_sampling.py](https://github.com/hstorm/pp_eaae_rennes/blob/master/examples/prior_sampling.py)

4. Potential outcome framework on numypro, including a verison using a  flexible dense neural network [neuralnet_treatment.py](https://github.com/hstorm/pp_eaae_rennes/blob/master/examples/neuralnet_treatment.py)

[Other examples, not considered in keynote talk]

5. Logit model implemented with numpyro neural network tools [logit_as_neuralnet.py](https://github.com/hstorm/pp_eaae_rennes/blob/master/examples/logit_as_neuralnet.py)

6. Market model (example incomplete!) [market_model.py](https://github.com/hstorm/pp_eaae_rennes/blob/master/examples/market_model.py)

## EAAE Poster: Probabilistic Programming policy application 

Kuhn, K., Pahmeyer, C., **Storm, H.** (2023) *Using Probabilistic Programming to Asses Policy Induced Adaptation of Crop Choices: A Case Study of the German Implementation of the EU Nitrates Directive*, Poster presented at EAAE congress 2023, Rennes, France. [Link Poster](https://github.com/hstorm/pp_eaae_rennes/blob/master/RedAreaPoster.pdf)

## Notes to setup environment using pipenv or vscode devcontainer

1) Install pipenv dependencies ```pipenv install --dev```

2) In order to use GPU support it is required to install numpyro[cuda] manually using wheels, currently seems to be not supported in pipenv

    ```
    pipenv shell

    pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```
    
    Optionally install flax manually:
    ```pip install flax```

[Alternativly use VS Code devcontainer](https://code.visualstudio.com/docs/remote/containers): using provided docker file and devcontainer.json (jax and flax need to be manually installed in container as with pipenv)