import subprocess

# alphas  = [1,0,0, 0.9,0.9, 0.8,0.8,0.8,  0.6,0.6,0.6, 0.5,0.5,0.0]
# betas   = [0,1,0, 0.1,0.0, 0.2,0.0,0.1,  0.4,0.0,0.2, 0.5,0.0,0.5]
# gammas  = [0,0,1, 0.0,0.1, 0.0,0.2,0.1,  0.0,0.4,0.2, 0.0,0.5,0.5]

alphas  = [1, 0.9, 0.8,0.8,  0.6,0.6, ]
betas   = [0, 0.1, 0.2,0.1,  0.4,0.2, ]
gammas  = [0, 0.0, 0.0,0.1,  0.0,0.2, ]
for num_epochs in [20,40,60]:
    print("num_epochs: ", num_epochs)
    for alpha, beta, gamma in zip(alphas, betas, gammas):
        # run train.py with alpha, beta, gamma by using subprocess
        subprocess.run(["python", "train.py", "--alpha", str(alpha), "--beta", str(beta), "--gamma", str(gamma), "--num_epochs", str(num_epochs)])
        # print("-------------------")