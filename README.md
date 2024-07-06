# Catch the Fruits - NEAT AI Game

## Overview

"Catch the Fruits" is a game built with Python using the Pygame library, where a player-controlled character catches falling fruits while avoiding bombs. The game employs NEAT (NeuroEvolution of Augmenting Topologies) to evolve an AI that learns to play the game effectively.

## Table of Contents

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Running the Game](#running-the-game)
4. [Training the AI](#training-the-ai)
5. [Using the Trained AI](#using-the-trained-ai)
6. [Files and Directories](#files-and-directories)

## Requirements

- Python 3.6+
- Pygame
- NEAT-Python

## Setup

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/Dramocrystal/catch-the-fruits.git
    cd catch-the-fruits
    ```

2. Install the required Python packages:
    ```bash
    pip install pygame neat-python
    ```

3. Ensure the following directory structure:
    ```
    catch-the-fruits/
    ├── Assets/
    │   ├── player.png
    │   ├── bg.png
    │   ├── fruit0.png
    │   ├── fruit1.png
    │   ├── fruit2.png
    │   ├── fruit3.png
    │   ├── fruit4.png
    │   └── bomb.png
    ├── config.txt
    ├── main.py
    └── winner.pkl
    ```
