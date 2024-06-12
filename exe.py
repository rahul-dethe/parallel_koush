import time
import sys
from setup import readInput
from MCCI import performMCCI

from mpi4py import MPI
subroutine = MPI.COMM_WORLD
size = subroutine.Get_size()
rank = subroutine.Get_rank()

start = time.time()

model, nSite, subSpace, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder, outputfile, restart, saveBasis  = readInput()

if rank == 0:
    newline = ("\nTotal Posible Determinats are %d .\nBreakup are [Ms, No of Determinants] - ")% (sum(posibleDet))
    with open (outputfile, "a") as fout:
        fout.write(newline)
        
    for i in range(len(Ms)):
        newline = ("\t[%d, %d]")%(Ms[i], posibleDet[i]) 
        with open(outputfile, "a") as fout:
            fout.write(newline)
            if (i+1 == len(Ms)):
                fout.write("\n\n")

    if ( subSpace > (sum(posibleDet) *0.8)):
        sys.exit("Sub-Space size is more than 80 % of total determinants space. Make Sub-Space size smaller and run it again.\n ")

performMCCI()

if rank == 0:
    newline = ("Total Time Taken in MCCI Calculation is %f sec.")%( time.time() - start )
    with open(outputfile, "a") as fout:
        fout.write(newline)
MPI.Finalize()