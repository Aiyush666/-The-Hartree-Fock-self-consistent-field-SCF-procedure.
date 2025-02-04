# -*- coding: utf-8 -*-
"""Aiyush_HF_code """

import math
import numpy as np
import matplotlib.pyplot as plt

#Read the nuclear repulsion energy from the enuc.dat

with open("enuc.dat",'r') as input_file:

    repulsion_energy=float(input_file.readline().strip())

print(f"The Nuclear repulsion Energy:{repulsion_energy}")

#Reading the overlap(s.dat),Electronic Kinetic Energy(t.dat),Electronic Potential Energy(v.dat)

with open("s.dat", 'r') as overlap:
    content1 = overlap.read()

with open ("t.dat",'r') as Kinetic_Energy:
    content2= Kinetic_Energy.read()

with open ("v.dat",'r') as Potential_Energy:
    content3=Potential_Energy.read()


# Define the size of the matrix based on the maximum index (in this case 7x7)
# Function to read the data from file and construct a symmetric matrix
def read_matrix(data, n_basis):
    # Initialize the matrix
    matrix = np.zeros((n_basis, n_basis))

    # Populate the matrix based on the data
    for line in data:
        # Split each line into indices and values
        li = line.split()

        i = int(li[0]) - 1  # Adjust for zero-based index
        j = int(li[1]) - 1
        value = float(li[2])

        # Assign the value to the matrix
        matrix[i][j] = value

        # Since the matrix is symmetric, fill the corresponding symmetric entry
        if i != j:
            matrix[j][i] = value

    return matrix

# Function to compute the core Hamiltonian (H_core = T + V)
def form_core_hamiltonian(kinetic, nuclear):
    return kinetic + nuclear

# Example data for kinetic energy matrix (fke or s.dat)
kinetic_data = open("t.dat",'r')

# Example data for nuclear attraction matrix (v.dat)
nuclear_data = open("v.dat",'r')

#Example data for overlap matrix(s.dat)
s_data=open("s.dat",'r')


# Set the number of basis functions
n_basis = 7

# Read the kinetic energy matrix (T)
kinetic_matrix = read_matrix(kinetic_data, n_basis)
np.set_printoptions(precision=7)

# Read the nuclear attraction matrix (V)
nuclear_matrix = read_matrix(nuclear_data, n_basis)
np.set_printoptions(precision=7)

#Read the Overlap matrix (V)
overlap_matrix=read_matrix(s_data,n_basis)
np.set_printoptions(precision=7)

# Form the core Hamiltonian (H_core = T + V)
core_hamiltonian = form_core_hamiltonian(kinetic_matrix, nuclear_matrix)
np.set_printoptions(precision=7)

# Print the resulting matrices
print(f"Kinetic Energy Matrix (T):\n", )
def print_mat(A,m,n):
  for i in range(m):
    for j in range(n):
      print("%12.6f"  %A[i,j], end="")
    print(end="\n")
print_mat(kinetic_matrix,7,7)
print("\nNuclear Attraction Matrix (V):\n")
def print_mat(A,m,n):
  for i in range(m):
    for j in range(n):
      print("%12.6f"  %A[i,j], end="")
    print(end="\n")
print_mat(nuclear_matrix,7,7)
print("overlap Matrix:\n")
def print_mat(A,m,n):
  for i in range(m):
    for j in range(n):
      print("%12.6f"  %A[i,j], end="")
    print(end="\n")
print_mat(overlap_matrix,7,7)
print(f"\nCore Hamiltonian (H_core = T + V):\n")
def print_mat(A,m,n):
  for i in range(m):
    for j in range(n):
      print("%12.6f"  %A[i,j], end="")
    print(end="\n")

print_mat(core_hamiltonian,7,7)

#Reading Electron repulsion Integrals:

#define maximum number of the basics function:

n_integrals=406

#Initialize a one dimonsional array to store the eri values

eri=np.zeros((n_integrals))

def compound_index(mu,nu,lam,sigma):

    if mu < nu:
        mu,nu=nu,mu
    elif lam < sigma:
        lam,sigma=sigma,lam
    elif (mu,nu) < (lam,sigma):
        mu,nu,lam,sigma=sigma,lam,mu,nu

    #calculate indices for unique storage:
    b= mu * (mu+1) // 2+ nu
    c= lam * (lam+1) // 2+ sigma

    if b >= c:
        return b * (b + 1) // 2 + c
    else:
        return c * (c + 1) // 2 + b

with open("eri.dat",'r') as Electron_repulsion_Integrals:

    for line_1 in Electron_repulsion_Integrals:
        lk=line_1.split()
        mu, nu, lam, sigma=int(lk[0])-1,int(lk[1])-1,int(lk[2])-1,int(lk[3])-1
        value=float(lk[4])

        index=compound_index(mu,nu,lam,sigma)
        eri[index]=value

#print("The  Electron Repulsion Integrals:\n",eri[index])

#Eigenvalues and Eigenvectores of an overlap matrix:

s_eigenvalue,s_eigenvectores=np.linalg.eigh(overlap_matrix)

#Diagonalization of the overlap matrix:
s_diagonalize=np.zeros((n_basis,n_basis))

 #S^(-1/2) Matrix
for i in range(7):
  s_diagonalize[i][i] = 1/(math.sqrt(s_eigenvalue[i]))

s_sqrt_inv = np.zeros((7,7))
a = np.matmul(s_diagonalize,s_eigenvectores.transpose())
s_sqrt_inv = np.matmul(s_eigenvectores,a)

print(f"\n The S^-1/2 is:\n")
def print_mat(A,m,n):
  for i in range(m):
    for j in range(n):
      print("%12.6f"  %A[i,j], end="")
    print(end="\n")
print_mat(s_sqrt_inv,7,7)


#Initial density Matrix Construction


Fock_matrix= np.zeros((7,7))

# Matrix Multiplication building the Fock Matrix:

A=np.matmul(core_hamiltonian,s_sqrt_inv)
Fock_matrix=np.matmul(s_sqrt_inv.transpose(),A)
print('\nTransformed Fock matrix = ')
print_mat(Fock_matrix,7,7)

#Eigenvectors transform to the original non-orthogonal AO basics:

orb_eng, Cp = np.linalg.eigh(Fock_matrix)
C0 = np.zeros((7,7))
C0 = np.matmul(s_sqrt_inv,Cp)
print('Coeffients = ')
print_mat(C0,7,7)
# Density Matrix

D = np.zeros((7,7))
for i in range(7):
  for j in range(7):
    for k in range(5):
      D[i][j]+=C0[i][k]*C0[j][k]


print("Density Matrix \n:")
def print_mat(A,m,n):
  for i in range(m):
    for j in range(n):
      print("%12.6f"  %A[i,j], end="")
    print(end="\n")
print_mat(D,7,7)


# Computing SCF Energy

E_elec = 0.0
for i in range(7):
  for j in range(7):
    E_elec+= D[i][j]*(core_hamiltonian[i][j]) + D[i][j]*(core_hamiltonian[i][j])   # SCF electronic energy calculated from the density matrix

E_tot0 = E_elec + repulsion_energy
print("Electronic Energy:",E_elec)

# Creating the New Fock Matrix

def cpd_index(a,b):
  if a >= b:
    return a*(a+1)/2 + b
  else :
    return b*(b+1)/2 + a

def new_F(D,eri):
    Fnew = np.zeros((7,7))
    for i in range(7):
      for j in range(7):
        Fnew[i][j] = core_hamiltonian[i][j]
        for k in range(7):
          for l in range(7):
            ij = cpd_index(i,j)
            ik = cpd_index(i,k)
            kl = cpd_index(k,l)
            jl = cpd_index(j,l)
            ijkl = int(cpd_index(ij,kl))
            ikjl = int(cpd_index(ik,jl))
            Fnew[i][j] += D[k][l]*(2.0*eri[ijkl] - eri[ikjl])                  # Building the new Fock matrix from D matrix of previous iteration

    return Fnew

  # Making new density matrix

def new_D(Fnew):
    a = np.matmul(Fnew,s_sqrt_inv)
    Fi = np.matmul(s_sqrt_inv.transpose(),a)



    orb_eng  , Cp= np.linalg.eigh(Fi)                                          # Cp (the coefficients) is the coefficient of the eigenvector (C0 prime).
    C0 = np.zeros((7,7))
    C0 = np.matmul(s_sqrt_inv,Cp)
    """
    print('C'':')
    print(Cp)
    print('C0:')
    print(C0)
    """

    Di = np.zeros((7,7))
    for i in range(7):
      for j in range(7):
        for k in range(5):
          Di[i][j]+=C0[i][k]*C0[j][k]                                          # Building i-th iteration Density Matrix

    return Di



# Computing the new SCF Energy

def new_e(hc,D,Fnew):
    e_eleci = 0

    for i in range(7):
      for j in range(7):
        e_eleci += D[i][j]*(hc[i][j] + Fnew[i][j])                             # Electronic energy on i-th iteration

    e_toti = e_eleci + repulsion_energy                                                # Total energy on i-th iteration
    return e_toti , e_eleci

# Testing for Convergence

e_toti_prev = E_tot0

delta1 = 1e-12
delta2 = 1e-11
ctr = 0
print("%s    %s            %s          %s        %s" %('Iter','E(elec)','E(tot)','Delta(E)','RMSD(E)'))
print("%2d %15.12f %15.12f" %(ctr,E_elec,E_tot0))

while True :
  Fnew = new_F(D,eri)
  Di = new_D(Fnew)
  e_toti , e_eleci = new_e(core_hamiltonian,Di,Fnew)

  delE = e_toti - e_toti_prev
  RMSD = 0
  for i in range(7):
    for j in range(7):
      RMSD += (Di[i][j] - D[i][j])*(Di[i][j] - D[i][j])

  RMSD = math.sqrt(RMSD)
  ctr+=1
  print("%2d %15.12f  %15.12f  %15.12f %13.12f" %(ctr,e_eleci,e_toti,delE,RMSD))
  if abs(delE) < delta1 and RMSD < delta2 :                                    # Comparing the energy difference and the Root-Mean-Squared-Difference to respective thresholds for convergence
      e_elec_final=e_eleci
      break
  e_toti_prev = e_toti
  D = Di

print("\n")
print("Electronic Energy converged at : E(elec) = ",(e_elec_final)," Hartrees")