
#information that is provided
#percentage present in whold population
prob_X = 0.03
prob_not_X = 1 - prob_X

prob_not_Y_given_X = 0.01
prob_Y_given_not_X = 0.10

prob_Y_given_X = 1 - prob_not_Y_given_X

#accounting for all Y - those Y for X and Y for not X
prob_Y = (prob_Y_given_X * prob_X) + (prob_Y_given_not_X * prob_not_X)

prob_X_given_Y = (prob_Y_given_X * prob_X) / prob_Y


print(prob_X_given_Y)


