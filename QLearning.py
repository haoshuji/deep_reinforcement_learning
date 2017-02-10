import numpy as np
import argparse

def Q_learning(R_matrix, gamma, num_iter):
    Q_matrix = np.zeros((R_matrix.shape[0], R_matrix.shape[0]))
    assert R_matrix.shape == Q_matrix.shape
    
    state_val_actions = {}
    for i in range(Q_matrix.shape[0]):
        state_val_actions[i]  = (np.where(R_matrix[i] >= 0)[0]).tolist()
    print state_val_actions

    print('Beginning to train')
    num_states = Q_matrix.shape[0]
    for i in range(num_iter):
        print('{} iteration'.format(i))
        start_state = np.random.randint(num_states)
        next_state = start_state
        while next_state != num_states-1:
            current_state = next_state
            num_possible_actions = len(state_val_actions[current_state])
            action = state_val_actions[current_state][np.random.randint(num_possible_actions)]
            next_state = action
            max_q_next_state = max([Q_matrix[next_state, each_action] for each_action in state_val_actions[next_state]])
            Q_matrix[current_state, action] = R_matrix[current_state, action] + gamma * max_q_next_state

        print Q_matrix
        
    return Q_matrix

def Test_Q(Q_matrix):
    for i in range(Q_matrix.shape[0]):
        print('{} test case'.format(i))
        path = [i]
        current_state = i
        next_state = current_state
        while next_state != Q_matrix.shape[0] - 1:
            current_state = next_state
            next_state = np.argmax(Q_matrix[current_state])
            path.append(next_state)
        print path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-r', '--reward_file', dest='reward_file', type=str,
                        default='./reward.csv', help='reward file name')
    parser.add_argument('-g', '--gamma', dest='gamma', type=float,
                        default=0.8, help='learning rate for q-learning')
    parser.add_argument('-i', '--num_iter', dest='num_iter', type=int,
                        default=20, help='number of interation to train q-learning')
    args = parser.parse_args()
    print args.reward_file
    R_matrix = np.loadtxt(args.reward_file, delimiter=',')
    print('There are {} state'.format(R_matrix.shape[0]))

    Q_matrix = Q_learning(R_matrix, args.gamma, args.num_iter)
    
    Q_matrix = Q_matrix/np.max(Q_matrix)

    Test_Q(Q_matrix)

        

