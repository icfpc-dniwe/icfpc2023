```
    ___    _  _     ___  __      __ ___               _       _
   |   \  | \| |   |_ _| \ \    / /| __|     o O O   (_)     (_)      o      __ _
   | |) | | .` |    | |   \ \/\/ / | _|     o         _       _      o      / o  |
   |___/  |_|\_|   |___|   \_/\_/  |___|   TS__[O]  _(_)_   _(_)_   TS__[O] \__,_|
 _|"""""|_|"""""|_|"""""|_|"""""|_|"""""| {======|_|"""""|_|"""""| {======|_|"""""|
 "`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'./o--000'"`-0-0-'"`-0-0-'./o--000'"`-0-0-'

                                 Chasing bottoms!
```

## Intro

This was a heck of a year, and most of the work was done by Vuvko.

This year's problem is similar to various optimization tasks, such as facility placement and maximum flow. The challenge with utilizing established solutions is the issue of blocking, making the problem particularly interesting.

## Diving in

Right from the start, I had a hunch that I had encountered similar tasks as an established optimization problem. Initially overlooking the blocking rule (since who reads the specifications on the first try?), I quickly devised a way to compose a MILP solver. Unfortunately, when I attempted to apply it to problem #1, it proved to be too demanding for my RAM to handle. This, coupled with re-reading the specifications, led me to the idea of using a greedy algorithm.

### Getting greedy

I had a codebase that could efficiently calculate the cost matrix for each potential placement (previously sampled) and each instrument. This enabled me to obtain an optimal placement for a musician playing a specific instrument. By iterating through all possible musicians (not necessarily in the given order), I obtained initial solutions.
### Blockers

At this point, the only thing missing was a solution to recalculate my cost matrix, considering the musicians already placed. Once I achieved that, I made modifications to the greedy algorithm. However, the recalculations after each step proved to be rather slow. Consequently, a new heuristic was introduced.
### Giving it space

While working with my solver, I noticed that the cost matrix wasn't changing as much as I had anticipated. Therefore, I could afford to recalculate it periodically to accelerate the simulation. Eventually, I ended up with a fast solver that performed impressively in a lightning round, propelling our team to a commendable #16 position on the leaderboard. Although I experimented with other tweaks and heuristics, they didn't yield substantial improvements.
## Learning to place

Being a fan of AI, I always endeavor to explore new things like Reinforcement Learning (RL) each year. This year marked my successful creation of a custom environment within the OpenAI Gym framework. I conducted numerous experiments using DeepMind's Dreamer and models from StableBaselines3.

It was an immensely enjoyable experience. However, none of the approaches worked as well as the modified greedy solver, and it took hours for a model to learn a given task.
### Thoughts about RL

The design of the environment and the reward function proved to be crucial elements. In my initial design, the models would converge to collapse, producing identical outputs at every simulation step. Upon investigating the issue, it became evident that I needed to alter my environment.

Firstly, I had to ensure that the reward function genuinely rewarded correct placements of musicians, emphasizing the importance of placing everyone. Secondly, the reward's magnitude and the position values needed to be scaled down, as neural networks struggle when dealing with large numbers (either as input or output).

Although these adjustments were helpful, they didn't bring about a significant transformation. Models still tended to converge on the same move, which troubled me. In my environment, almost every observation remained fixed, with the model only perceiving slight changes as it navigated through the environment. This realization prompted a series of changes in the environment's design. Voila! The models now genuinely attempted to place musicians in different positions. Although it worked better than random placement, it was inferior to the greedy algorithm.
### Custom network

As a side exploration, I experimented with a feature extraction model, aiming to incorporate additional knowledge about the task into the RL model. However, the results were not as impressive as I had anticipated.
## Expansions

The first two expansions were introduced, but they didn't significantly impact my solution. They could be easily incorporated into the calculation of the cost matrix. After implementing them and uploading solutions for every new problem, I returned to my RL experiments. Despite their drawbacks of being slow and somewhat unintelligent, I genuinely enjoy working with neural networks.

The final expansion, however, changed everything. I could now disable any musician that would lower my best score in the greedy algorithm. And that's it. No further modifications to the heuristics or the solution were required.
## Side notes

I attempted to utilize ChatGPT, but I couldn't obtain anything truly useful from it. It seemed to attempt a brute force approach to the problem, which was highly inefficient.

However, I was surprised by its ability to write a simulation almost flawlessly.
