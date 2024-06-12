from mpi4py import MPI, Rc, rc
Rc.initialize = True
rc.initialize = True
subroutine = MPI.COMM_WORLD
size = subroutine.Get_size()
rank = subroutine.Get_rank()

def s2(basis, vecm, sProduct):
    chunks = int(len(vecm) // size)
    start = rank * chunks    
    end = 0
    if rank < size - 1:        
        end = start + chunks
    else:
        end = len(vecm)

    sSquare = 0
   
    for zz in range(start, end):
        s1Square = 0
        szVal = 0
        sxyVal = 0
        c1 = vecm[zz]
        c2 = 0

        for ix in sProduct:
            if(ix[0] == ix[1]):
                s1Square += (0.75)*c1*c1	
            else:
                if(basis[zz][ix[0]]) == (basis[zz][ix[1]]):
                    szVal += 0.25*c1*c1                
                else:
                    szVal -= 0.25*c1*c1
                    basis1 = list(basis[zz])
                    basis1[ix[0]], basis1[ix[1]] = basis1[ix[1]], basis1[ix[0]]
                    basis2 = ''.join(basis1)
                    if basis2 in basis:
                        basis1Key = basis.index(basis2)
                        c2 = vecm[basis1Key]
                        sxyVal += 0.5 *c1*c2
        sSquare += s1Square  + szVal  + sxyVal
    return round(sSquare, 4)

###########################################################


####################The End#################################
