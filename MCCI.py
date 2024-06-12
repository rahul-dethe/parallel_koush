import numpy as np
from bitstring import BitArray
import random
import math
import net_nstates
from newGeneration import makeNewGeneration, makeNewMlGeneration
from convergence import checkConvergence, checkFinalConv, makeFitGeneration, convInitializer, update, updateDeterminatList
from spinCalculator import spinCalculator, stateFinder
from setup import readInput

from mpi4py import MPI
subroutine = MPI.COMM_WORLD
size = subroutine.Get_size()
rank = subroutine.Get_rank()

model, nSite, subSpace, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder, outputfile, restart, saveBasis = readInput()

if model == 'HB':
    from HeisenHam import Hamiltonian
if model == 'GM':
    from GhoshMajumHam import Hamiltonian

mlStart = 4
mlPerSpace = 3
spaceIncrease = 1000

dataFile = outputfile + ".TrainData_subSpace.csv"

def performMCCI():    
    convReach = False   
    subBasis = []    
    if restart:
        with open(saveBasis,"r") as fsaveB:
            for i in range(subSpace):
                line = fsaveB.readline()
                det0 = BitArray(bin=line.strip())
                subBasis.append(det0)
    
    if (restart == False):        
        for i in range(len(det)):
            det0 = det[i]
            random.shuffle(det0)
            subBasis.append(det0)
            if (Ms[0] == 0):
                subBasis.append(~det0)
                detCopy0 = BitArray()
        for i in range(len(det)):
            while len(subBasis) < int(subSpace * (i + 1) /len(det)) :
                detCopy0 = subBasis[i *2].copy()
                random.shuffle(detCopy0)
                if detCopy0 not in list(subBasis):
                    subBasis.append(detCopy0)
                    if (Ms[0] == 0):
                        if ~detCopy0 not in list(subBasis):
                            subBasis.append(~detCopy0)
    
    """**************************************************************************************"""
    sh = Hamiltonian(subBasis)
    subHam = subroutine.allreduce(sh, op=MPI.SUM)
    """**************************************************************************************"""

    lenSB = len(subBasis)
    energy = np.zeros(lenSB)
    ciCoef = np.zeros((nStates * lenSB))

    net_nstates.diagonalization(hamil = subHam, n = lenSB, n1 = 3 * lenSB, n2 = nStates, ehamil = energy, vec = ciCoef)
    
    energyMin = energy[ 0 ]
    ciCoefMin = ciCoef[ 0 : lenSB]
    #print(energy[ 0 : nStates])

    s2ValMin = 100
    targetState,  s2ValDiff,  energyChange = ([], ) * 3
    targetState, s2ValList, s2ValDiff, energyChange, spinChange = convInitializer()
    
    ## to store all the det and their CI coef to a data file
    allDet = []
    allCicoef = []
    kValue = [0, 0]     # to check if space size increased or not
    for i in range(maxItr): # maxItr
        # for dynamic sub space size
        k = max(0, math.floor((i - mlStart - 3)/mlPerSpace))
        newSize = k * spaceIncrease + subSpace

        kValue[0] = kValue[1]
        kValue[1] = k

        kDiff = kValue[1] - kValue[0]

        # creation oif new generation
        if (i <= mlStart):
            newGen, lenNewGen = makeNewGeneration(subBasis)
        
        if (i == mlStart +1):
            if rank == 0:
                newline = ("\nStarting Active-Learning Protocal \n")
                with open(outputfile, "a") as fout:
                    fout.write(newline)
        
        if (i > mlStart):
            if rank == 0:
                newGen, lenNewGen, allDet, allCicoef = makeNewMlGeneration(subBasis, dataFile, newSize, allDet, allCicoef, k)
        
        """************************************************************************************"""
        newGen = subroutine.bcast(newGen, root=0)
        lenNewGen = subroutine.bcast(lenNewGen, root=0)
        allDet = subroutine.bcast(allDet, root=0)
        allCicoef = subroutine.bcast(allCicoef, root=0)
        """************************************************************************************"""

        # print("i, lenNewGen", i, lenNewGen)
        """**************************************************************************************"""
        ng = Hamiltonian(newGen)
        newGenHam = subroutine.allreduce(ng, op=MPI.SUM)
        # print("\nRank->", rank, newGenHam)
        """**************************************************************************************"""

        energy = np.zeros(lenNewGen )
        ciCoef = np.zeros(lenNewGen * nStates )
        net_nstates.diagonalization(hamil = newGenHam, n= lenNewGen, n1 = 3 * lenNewGen, n2 = nStates, ehamil = energy , vec = ciCoef)
    
        s2ValList =  spinCalculator(newGen, energy[ 0 : nStates ], ciCoef, lenNewGen, convReach)
        
        if (i < startSpinTargetItr):
            targetState[1] = 0
            s2ValDiff = [10, 10]    # dont want to inculde spin information on optimizations        
        
        if (i == startSpinTargetItr): # for smmoth transition from non spin target to spin target cacluations            
            targetState[1], s2ValDiff[1] = stateFinder(s2ValList,s2Target)  # for first state of a particular spin
            if rank == 0:
                newline = ("\nStarting Optimization W.R.T Spin, Target State Spin Value -> %f \n\n")%(s2Target)
                with open(outputfile, "a") as fout:
                    fout.write(newline)
            energyMin = energy[ targetState [ 1 ] ]
            s2ValDiff[0] = s2ValDiff[1]  
        
        if (i > startSpinTargetItr):
            targetState[1], s2ValDiff[1] = stateFinder(s2ValList,s2Target)  # for first state of a particular spin

        ciCoefNew = ciCoef[(lenNewGen) * targetState[1] : (lenNewGen) * (targetState[1] +1)]
        energyNew = energy[ targetState [ 1 ] ]
        s2ValNew = s2ValList [targetState [ 1 ]]   
        Eith = energyMin
        allDet,  allCicoef = updateDeterminatList(allDet, allCicoef, newGen, ciCoefNew, dataFile, kDiff)

        #print("yes1")
        subBasis, energyMin, ciCoefMin, s2ValDiff, s2ValMin, energyUpdate = checkConvergence( energyMin, energyNew, ciCoefMin, ciCoefNew, s2ValMin, s2ValNew, targetState,  newGen, s2ValDiff, i, newSize)

        #print("yes2")
        energyChange, spinChange, convReach = checkFinalConv( energyChange, spinChange,  Eith, energyMin, s2ValDiff[0], convReach) 
        
        """***************************************************************************************************"""
        energyUpdate = subroutine.bcast(energyUpdate, root=0)
        if energyUpdate :
            energyFinal, ciFinal, basisFinal = update( energy[0 : nStates], ciCoef, newGen, len(newGen) )
        """***************************************************************************************************"""
       
        if (convReach == True) or (i == maxItr - 1):
            if convReach:
                if rank == 0:
                    newline = ("\nIteration Converged.\n")
                    with open(outputfile, "a") as fout:
                        fout.write(newline)
            else:
                if rank == 0:
                    newline = ("\nReach Max Iteration Number.\n")
                    with open(outputfile, "a") as fout:
                        fout.write(newline)   
                convReach = True
            ## Final Calculation
            spinCalculator(basisFinal, energyFinal, ciFinal, len(basisFinal), convReach)   
            break
    

    if rank == 0:
        bF, cF = makeFitGeneration(basisFinal, ciFinal[: len(basisFinal)], len(basisFinal))   # for ordered
    #bF, cF = makeFitGenerationAL(basis_print, ci_print, lenNewGen)   # for ordered

        with open( (str(outputfile) + '.basis'), "w") as fbasis:
            for element in bF:
                fbasis.write(element.bin +'\n')

        with open( (str(outputfile) + '.ci'), "w") as fci:
            for element in cF:
                fci.write(str(round(float(element),6)) +'\n')

