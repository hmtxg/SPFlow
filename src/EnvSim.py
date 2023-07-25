import csv
import random

# Function to simulate a single stage game
def simulate_first_game(player1_action):
    opponentA = random.choice([0, 1])  # 0 represents defection, 1 represents cooperation
    if player1_action == 1 and opponentA == 1:
        reward = 10
    elif player1_action == 1 and opponentA == 0:
        reward = -10
    elif player1_action == 0 and opponentA == 1:
        reward = 20
    else:
        reward = 0

    return reward, opponentA

def TFTaction(reward):
    if reward == 10: #BOTH COOPERATE - CONTINUE TO COOPERATE
        player_action = 1
    if reward == 20: #P1 DEFECT, P2 COOPERATE - COOPERATE NEXT
        player_action = 1
    if reward == 0: #BOTH DEFECT - CONTINUE TO DEFECT
        player_action = 0
    if reward == -10: #P1 COOPERATE, P2 DEFECT - DEFECT NEXT
        player_action = 0
    
    return player_action

def getReward(playerA, opponentA):
    if playerA == 1 and opponentA == 1:
        reward = 10
    elif playerA == 1 and opponentA == 0:
        reward = -10
    elif playerA == 0 and opponentA == 1:
        reward = 20
    else:
        reward = 0
    return reward

def main():
    num_iterations = 50000
    playerA1 = 0 
    reward0 = 0
    with open('PD_3S_Mirror.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Stage0Reward', 'Stage1Action', 'Stage1Reward', 'Stage2Action', 'Stage2Reward', 'Stage3Action', 'Stage3Reward', 'AvgReward'])
        for _ in range(num_iterations): 
            reward1, opponentA1 = simulate_first_game(playerA1)
            playerA2 = TFTaction(reward1)
            reward2, opponentA2 = simulate_first_game(playerA2) #Opponent still acts randomly, or else defect loop
            playerA3 = TFTaction(reward2)
            opponentA3 = playerA2
            reward3 = getReward(playerA3, opponentA3)
            total_reward = reward1 + reward2 + reward3
            avg_reward = total_reward / 3
            writer.writerow([reward0, playerA1, reward1, playerA2, reward2, playerA3, reward3, avg_reward])

if __name__ == "__main__":
    main()