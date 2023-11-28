import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import argparse
import struct

np.random.seed(0)
def generateUpperTriangularCSR(n_rows, avg_elements_per_row):
    triplets = []
    ## generate random sparse matrix
    for i in range(n_rows):
        ## pick a random number of elements for this row
        for _ in range(int(avg_elements_per_row)):
            j = np.random.randint(n_rows)
            while j == i:
                j = np.random.randint(n_rows)
            ## make sure it is upper triangular
            if i < j:
                triplets.append((i, j, 1.0))
            else:
                triplets.append((j, i, 1.0))

    rows, cols, values = zip(*triplets)

    # Create a COO matrix from the triplets
    coo = coo_matrix((values, (rows, cols)))
    csr = coo.tocsr()
    return csr


def expandCSR(csr, block_size, n_rows):
    outerIndices = csr.indptr
    innerIndices = csr.indices
    rows = np.zeros(n_rows * block_size * block_size + outerIndices[-1] * block_size * block_size * 2).astype(int)
    cols = np.zeros(n_rows * block_size * block_size + outerIndices[-1] * block_size * block_size * 2).astype(int)
    values = np.zeros(n_rows * block_size * block_size + outerIndices[-1] * block_size * block_size * 2)
    offDiagonalBlocks = np.zeros(outerIndices[-1] * block_size * block_size)
    diagonalBlocks = np.zeros(n_rows * block_size * block_size)

    tripletCount = 0
    diagonalBlockCount = 0
    offDiagonalBlockCount = 0
    diagonalBlock = np.zeros((block_size, block_size))
    for i in range(n_rows):
        ## deal with last row since upper triangular off diagonal
        ## does not contain the last row
        row = i
        diagonalBlock = np.random.rand(block_size, block_size)
        diagonalBlock = diagonalBlock + diagonalBlock.T
        for k in range(block_size):
            for l in range(block_size):
                rows[tripletCount] = row * block_size + k
                cols[tripletCount] = row * block_size + l
                values[tripletCount] = diagonalBlock[k][l]
                tripletCount += 1
        diagonalBlocks[diagonalBlockCount * block_size * block_size : (diagonalBlockCount + 1) * block_size * block_size] = diagonalBlock.flatten().tolist()
        diagonalBlockCount += 1

    block = np.zeros((block_size, block_size))
    for i in range(len(outerIndices) - 1):
        row = i
        for j in range(outerIndices[i], outerIndices[i + 1]):
            col = innerIndices[j]
            ## we need to expand this block
            block = np.random.rand(block_size, block_size)
            for k in range(block_size):
                for l in range(block_size):
                    ## generate a random value
                    v = block[k][l]
                    rows[tripletCount] = row * block_size + k
                    cols[tripletCount] = col * block_size + l
                    values[tripletCount] = v
                    tripletCount += 1
                    ## add the transpose
                    rows[tripletCount] = col * block_size + l
                    cols[tripletCount] = row * block_size + k
                    values[tripletCount] = v
                    tripletCount += 1
            offDiagonalBlocks[offDiagonalBlockCount * block_size * block_size : (offDiagonalBlockCount + 1) * block_size * block_size] = block.flatten().tolist()
            offDiagonalBlockCount += 1

    ## now construct the expanded csr
    coo = coo_matrix((values, (rows, cols)))
    expandedCSR = coo.tocsr()
    return expandedCSR, offDiagonalBlocks, diagonalBlocks


def main():
    ## Set up the argument parser
    parser = argparse.ArgumentParser(description='Generate CSR storage for a sparse matrix.')
    parser.add_argument('n_rows', type=int, help='Number of rows in the matrix')
    parser.add_argument('avg_elements_per_row', type=int, help='Average number of elements per row')
    parser.add_argument('block_size', type=int, help='Size of each block in the matrix')
    ## generate random sparse matrix
    args = parser.parse_args()
    n_rows = args.n_rows
    avg_elements_per_row = args.avg_elements_per_row
    block_size = args.block_size
    blockCSR = generateUpperTriangularCSR(n_rows, avg_elements_per_row)
    print("Block CSR rows", blockCSR.shape[0])
    print("Upper triangular construction done")
    print(f"Result average elements per row: {len(blockCSR.indices) / n_rows}")
    ## now we need to generate the csr for full matrix where each element is a block
    ## as well as the diagonal block values, the off diagonal block values
    expandedCSR, offDiagonalBlocks, diagonalBlocks = expandCSR(blockCSR, block_size, n_rows)
    print("Expanded CSR construction done")
    print(f"Result average elements per row: {len(expandedCSR.indices) / (n_rows * block_size)}")
    ## construct a dense vector
    denseVector = np.random.rand(n_rows * block_size)
    # ## now we need to compute the matrix vector product
    # ## we do it in two ways
    # ## 1. using the expanded csr
    # ## 2. using the block csr
    # ## we first do it using the expanded csr
    expandedCSRVector = expandedCSR.dot(denseVector)
    # ## now do the block
    # result = np.zeros(n_rows * block_size)
    # for i in range(n_rows):
    #     ## get the block
    #     diagonalBlock = np.array(diagonalBlocks[i * block_size * block_size : (i + 1) * block_size * block_size]).reshape((block_size, block_size))
    #     blockVector = diagonalBlock.dot(denseVector[i * block_size : (i + 1) * block_size])
    #     result[i * block_size : (i + 1) * block_size] += blockVector
    # count = 0
    # for i in range(len(blockCSR.indptr) - 1):
    #     row = i
    #     for j in range(blockCSR.indptr[i], blockCSR.indptr[i + 1]):
    #         col = blockCSR.indices[j]
    #         offDiagonalBlock = np.array(offDiagonalBlocks[count * block_size * block_size: (count + 1) * block_size * block_size]).reshape((block_size, block_size))
    #         blockVector = offDiagonalBlock.dot(denseVector[col * block_size : (col + 1) * block_size])
    #         result[i * block_size : (i + 1) * block_size] += blockVector

    #         ## now do the transpose
    #         blockVector = offDiagonalBlock.T.dot(denseVector[row * block_size : (row + 1) * block_size])
    #         result[col * block_size : (col + 1) * block_size] += blockVector
    #         count += 1
    # print("Norm of Difference: ", np.linalg.norm(expandedCSRVector - result))
    
    np.savetxt('block_csr_inner.txt', blockCSR.indices, fmt='%i')
    np.savetxt('block_csr_outer.txt', blockCSR.indptr, fmt='%i')
    np.savetxt("diagonal_blocks.txt", diagonalBlocks, fmt='%f')
    np.savetxt("off_diagonal_blocks.txt", offDiagonalBlocks, fmt='%f')
    np.savetxt("dense_vector.txt", denseVector, fmt='%f')
    np.savetxt("expanded_csr_inner.txt", expandedCSR.indices, fmt='%i')
    np.savetxt("expanded_csr_outer.txt", expandedCSR.indptr, fmt='%i')
    np.savetxt("expanded_csr_values.txt", expandedCSR.data, fmt='%f')
    np.savetxt("dimensions.txt", np.array([n_rows, block_size, len(blockCSR.indptr), len(blockCSR.indices), len(expandedCSR.indptr), len(expandedCSR.indices), len(diagonalBlocks), len(offDiagonalBlocks)]), fmt='%i')
    np.savetxt("result.txt", expandedCSRVector, fmt='%f')
    


if __name__ == '__main__':
    main()