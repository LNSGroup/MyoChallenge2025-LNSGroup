# DIY Submission
This README explains how to submit solutions without relying on GitHub actions. For seasoned participants: This was the primary submission process from the [MyoChallenge 2022](https://github.com/ET-BE/myoChallengeEval).

## Prerequisites
<details closed>
<summary>Install Docker compiler</summary>

Install docker following the [instructions](https://docs.docker.com/get-docker/). Once installed, you can compile the docker containers for the 2 agents with the following scripts:

Note: Do not use `sudo` privileges, rather manage Docker as a [non-root user](https://docs.docker.com/engine/install/linux-postinstall/). Building the docker with root privileges might cause problems during the evalai submission.
</details>

<details closed>
<summary>Register an account on EvalAI for the team</summary>

Install EvalAI Command Line Interface (*evalai-cli*)
```bash
pip install "evalai>=1.3.13"
```

You might need to install evalai dependencies first:

```bash
sudo apt-get install libxml2-dev libxslt-dev
```

[⚠️ IMPORTANT ⚠️] Each team needs to be registered to obtain a specific token to identify it (see [instructions](https://evalai.readthedocs.io/en/latest/participate.html)). After registration, it is possible to add the EvalAI account token to via evalai-cli (full list of commands [here](https://cli.eval.ai/)) with the following command:
``` bash
# Register the tocken to identify your contribution
evalai set_token <your EvalAI participant token>
# Test that the registration was successful. MyoChallenge needs to be in the list returned
evalai challenges --participant
```

</details>

<details closed>
<summary>Clone this repository </summary>

Clone this repository to have access to all needed files:
```bash
# Clone the repository
git clone https://github.com/MyoHub/myochallenge_2025eval.git
# Enter into the root path
cd myochallenge_2025eval
# Install dependencies and tests
source ./setup.sh
```
</details>

## STEP 1: Train your model
The API to interface with the environment is explained in the MyoSuite [docs](https://myosuite.readthedocs.io/en/latest/).

More information on the training and customization are provided [here](./agent/TrainingPolicies.md)

<!-- For this challenge you might want to try the `myoChallengeSoccerP1-v0` for a quick test of training a policy (it should take ~2h on a regular laptop) and test the evaluation process. -->

## STEP 2: Customize Agent Script
We provide 4 templates to describe how the agent will communicate with the environment during the evaluation.
-  TableTennis - random ([agent_Tabletennis_random.py](../agent/agent_tabletennis_random.py))
-  Soccer - random ([agent_soccer_random.py](../agent/agent_soccer_random.py))

The random templates are very simple and can be used to understand the general structure of the submission template, as well as test if everything works correctly.

When you customize these files to submit with your preferred learning framework, is important to add the dependencies that you need to the appropriate requirements files: Either  [agent_random.txt](../requirements/agent_random.txt). These dependencies are automatically installed when building the docker container.
Make sure none of your dependencies defaults to GPU-usage, as the containers are evaluated on CPU-only instances.

Once you have finished customizing the scripts, testing between the agent and environment can be performed by using the scripts below:
- Table Tennis random `sh ./test/test_tabletennis_agent_random.sh`
- Soccer random `sh ./test/test_soccer_agent_random.sh`

Upon successful testing, it is possible to submit the solution following next steps.

## STEP 3: Build a docker container with the agent
The evaluation will be based on the model submitted as a docker container. It is possible to build the docker in two ways with either directly docker build (Suggested method) or with docker-compose (Alternative method, this will require to install [docker-compose](https://docs.docker.com/compose/install/))
<details open>
<summary>Suggested Method: Using `docker build`</summary>

``` bash
# Compile the container for the Table Tennis Agent
docker build -f docker/agent/Dockerfile_AgentTabletennis_random . -t myochallengeeval_tabletennis_agent

# Compile the container for the Soccer Agent
docker build -f docker/agent/Dockerfile_AgentSoccer_random . -t myochallengeeval_soccer_agent
```
</details>

<details close>
<summary>Alternative Method: Using `docker-compose`</summary>


``` bash
# Compile the container for the Table Tennis Agent
docker-compose -f docker-compose-TabletennisAgent.yml up --build

# Compile the container for the Soccer Agent
docker-compose -f docker-compose-SoccerAgent.yml up --build
```
</br>
</details>

## Step 4: Upload the docker container on evalAI for evaluation

Push the docker image to [EvalAI docker registry](https://eval.ai/web/challenges/challenge-page/2373/submission) (it is possible to get the information about the image and TAG with the command `docker images`)

```bash
evalai push <image>:<tag> --phase <phase_name>
```
Use --private or --public flag in the submission command to make the submission private or public respectively.

for example, commands to upload agents for Phase 1 might look like (you can find those on evalai submission page):
- TableTennis Agent : `evalai push myochallengeeval_mani_agent:latest --phase myochallenge2025-XXXXX1-XXXX --public`

- Soccer Agent: `evalai push myochallengeeval_loco_agent:latest --phase myochallenge2025-XXXXXX2-XXXX --public`

<!-- and, for Phase 2 might look like:

- Table Tennis Agent : `evalai push Dockerfile_Mani:latest --phase myochallenge2023-maniphase2-2105 --public`

- Soccer Agent: `evalai push Dockerfile_Loco:latest --phase myochallenge2023-locophase2-2105 --public`
 -->

For more commands, please refer to [evalai-cli documentation](https://cli.eval.ai/) for additional commands.
