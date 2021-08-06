import emAlgorInference as full

def main():
    #number of controller nodes
    internalnodes = 2

    #initial distribution
    initial = full.randdistr(internalnodes)

    #baseling, normally set to false
    base = False

    #minimal number of iterations
    iterations = 1000

    # distance between the probability of success before stopping
    tol = 1*10**-5
    # output path
    path = '/home/carlotta/Documents/Tests/ResultsGoale5e5test'


    #full distribution
    results1 = full.emAlg(initial, internalnodes, iterations, tol, 0, base)

    #no reactive control
    results2 = full.emAlg(initial, internalnodes, iterations, tol, 1, base)

    #no information flow to the controller
    results3 = full.emAlg(initial, internalnodes, iterations, tol, 2, base)


    #write output
    output = open(path + '.txt', 'w')
    output.write("Full = np.array([")
    for i in range(27):
        output.write('[')
        my_string = ','.join(map(str, results1[i]))
        output.write(my_string)
        output.write('],')
    output.write('])')
    output.write('\n' )

    output.write("NoMorph = np.array([")
    for i in range(27):
        output.write('[')
        my_string = ','.join(map(str, results2[i]))
        output.write(my_string)
        output.write('],')
    output.write('])')
    output.write('\n' )

    output.write("NoCaus = np.array([")
    for i in range(27):
        output.write('[')
        my_string = ','.join(map(str, results3[i]))
        output.write(my_string)
        output.write('],')
    output.write('])')
    output.write('\n' )

    output.close()
main()
