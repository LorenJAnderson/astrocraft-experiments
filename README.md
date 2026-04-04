# AstroCraft Experiments

<img src="logo.JPG" width="200"> 

This repository provides the code for reproducing the experiments in the published paper "*Overcoming Challenges of Realism in Competitive Space-Based Reinforcement Learning with AstroCraft*" (citation below). For further information on the AstroCraft environment, see the [partner github repository]([https://github.com/LorenJAnderson/astrocraft-experiments](https://github.com/LorenJAnderson/astrocraft-environment).

## Running the Experiments
The ```comparisons``` folder contains the code for generating random games of other similar environments in the literature and provides the file to visualize and analyze the data (see the data table in the paper). The ```figures``` folder contains all necessary code to reproduce the figures of results in the paper, given the data. The ```offlinemodels``` folder and ```train.py``` file contain the code for training the model (see paper for complete training details). The ```graph_data_generation.ipynb``` file contains the code necessary to test the performance of the trained model, while the ```ppo_eval.py``` file gathers the same results for the trained proximal policy optimization (PPO) model for comparison purposes. Run the latter first. 

---

# Distribution Statements

Code: DISTRIBUTION A: Approved for public release; distribution is unlimited. Public Affairs release approval #AFRL-2023-3267.

Paper: DISTRIBUTION A: Approved for public release; distribution is unlimited. Public Affairs release approval #AFRL-2026-0901.

# Citation

Note -- As of the current commit, the paper has not been indexed by Google Scholar or IEEE Xplore. Once that happens, the below citation might change (est. June-July 2026). 

```
@inproceedings{anderson2026overcoming,
  title={Overcoming Challenges of Realism in Competitive Space-Based Reinforcement Learning with AstroCraft},
  author={Anderson, Loren J and Erwin, R Scott and Hansen, Moses and Outkin, Victoria and Qureshi, Rehman S and Aborizk, Anthony and Kulkarni, Rohan and Dennery, Elyssa and Senapathy, Sahitya and Makkapati, Srija and Flathmann, William and Alcade, Kathryn and Dokmeci, Berkan and Firouzbakht, Shahin and Basan, Edfil},
  booktitle={2026 IEEE Aerospace Conference},
  pages={1--13},
  year={2026},
  organization={IEEE}
}
