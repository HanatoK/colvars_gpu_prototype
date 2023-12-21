#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn as nn


class RMSD(nn.Module):
    def __init__(self, reference_file: str):
        super().__init__()
        reference_positions = np.loadtxt(reference_file).reshape((-1, 3))
        self.num_atoms = len(reference_positions)
        self.reference_positions, _ = self.bring_to_center(
            torch.tensor(
                reference_positions, dtype=torch.float64, requires_grad=True, device='cuda'))

    def bring_to_center(self, data):
        center_of_geometry = data.mean(dim=0)
        return (data - center_of_geometry), center_of_geometry

    def build_matrix_F(self, pos_target, pos_reference):
        mat_R = torch.matmul(pos_target.T, pos_reference)
        F00 = mat_R[0][0] + mat_R[1][1] + mat_R[2][2]
        F01 = mat_R[1][2] - mat_R[2][1]
        F02 = mat_R[2][0] - mat_R[0][2]
        F03 = mat_R[0][1] - mat_R[1][0]
        F10 = F01
        F11 = mat_R[0][0] - mat_R[1][1] - mat_R[2][2]
        F12 = mat_R[0][1] + mat_R[1][0]
        F13 = mat_R[0][2] + mat_R[2][0]
        F20 = F02
        F21 = F12
        F22 = -mat_R[0][0] + mat_R[1][1] - mat_R[2][2]
        F23 = mat_R[1][2] + mat_R[2][1]
        F30 = F03
        F31 = F13
        F32 = F23
        F33 = -mat_R[0][0] - mat_R[1][1] + mat_R[2][2]
        row0 = torch.stack((F00, F01, F02, F03))
        row1 = torch.stack((F10, F11, F12, F13))
        row2 = torch.stack((F20, F21, F22, F23))
        row3 = torch.stack((F30, F31, F32, F33))
        F = torch.stack((row0, row1, row2, row3))
        return F

    def rmsd_impl(self, atom_pos):
        atom_pos_centered, _ = self.bring_to_center(atom_pos)
        matrix_F = self.build_matrix_F(atom_pos_centered, self.reference_positions)
        w, v = torch.linalg.eigh(matrix_F)
        max_eig_val = w[-1]
        s = torch.sum(torch.square(atom_pos_centered) + torch.square(self.reference_positions))
        return torch.sqrt((s - 2.0 * max_eig_val) / self.num_atoms)

    @torch.jit.export
    def calc_value(self, atom_pos):
        return self.rmsd_impl(atom_pos)

    @torch.jit.export
    def calc_gradients(self, atom_pos):
        rmsd = self.rmsd_impl(atom_pos)
        grad = torch.autograd.grad([rmsd], [atom_pos])[0]
        if grad is None:
            return torch.zeros_like(atom_pos, dtype=torch.float64, device='cuda')
        else:
            return grad

    @torch.jit.export
    def apply_force(self, atom_pos, f):
        rmsd = self.rmsd_impl(atom_pos)
        grad = torch.autograd.grad([rmsd], [atom_pos])[0]
        if grad is None:
            return torch.zeros_like(atom_pos, dtype=torch.float64, device='cuda')
        else:
            return -1.0 * f * grad


m = torch.jit.script(RMSD('reference_frame.txt'))

torch.jit.save(m, 'RMSD.pt')
