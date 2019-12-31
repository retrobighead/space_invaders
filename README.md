## はじめに

## Configuration

### Download this repository



### pyenv / pyenv-virtualenv

pyenv / pyenv-virtualenv のインストール

```bash
# pyenv / pyenv-virtualenv のインストール (Homebrew を使用している場合)
$ brew install pyenv  
$ brew install pyenv-virtualenv

# PATH を通す
# zsh の場合 ~/.zprofile に書き込む
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
$ echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
$ echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile

# ./bash_profile を再読み込み
$ source ~/.bash_profile # ~/.zprofile
```

python のインストールと仮想環境の生成

```bash
# インストール可能なPythonのバージョン一覧
$ pyenv install --list

# 特定バージョンのPythonをインストール
$ pyenv install 3.6.6
$ pyenv versions # 確認

# 仮想環境の生成
$ pyenv virtualenv 3.6.6 space_invaders
$ pyenv versions # 確認

# ディレクトリ以下を仮想環境の管理化に置く (.python-versionの生成)
$ pyenv local space_invaders
$ pyenv versions # 確認
$ python --version # 確認 => 3.6.6
```

仮想環境の起動/終了

### pip によるパッケージ管理

- jupyter lab
- gym
- stable-baselines[mpi]==2.8.0
- torch

```bash
# pip のアップグレード
$ pip install -U pip

# miscellaneous
$ pip install tqdm

# jupyter lab
$ pip install jupyter
$ pip install jupyterlab

# OpenAI Gym のインストール
$ pip install gym
$ pip install gym[atari] # gym[all]

# stable-baselines のインストール
$ brew install cmake openmpi
$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.14.0-py3-none-any.whl
$ pip install stable-baselines[mpi]==2.8.0

# PyTorch のインストール
$ pip install torch torchvision
```
