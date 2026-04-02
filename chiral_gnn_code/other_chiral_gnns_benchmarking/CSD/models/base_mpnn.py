'''
Base CSD MPNN.
'''

import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"

class MPNN(nn.Module):
    '''
    Edge based message passing neural network. Messages are passed via a neural network
    based on the edge type and updated via a GRU.
    ----
    Args:
        message_size (int):
            The size of the message (zeros padding added as needed). This must be greater than or
            equal to the number of atomic features.

        message_passes (int):
            The number of times to run the message passing.

        ranked_unique_atoms (list):
            The list of atoms (atomic numbers) in decreasing order of commonness.

    Forward Args:
        g (tensor):
            The 7D array of the adjacency matrices based on edge type.
            Has size batch size x number of bond types x longest molecule x longest molecule.

        h (tensor):
            The initial atomic features for each atom.
            H has size batch size x longest molecule x atomic features length

    Returns:
        output (tensor):
            The embedded molecule.
    '''

    def __init__(self, message_size, message_passes, ranked_unique_atoms):
        super(MPNN, self).__init__()

        self.message_passes = message_passes
        self.message_size = message_size
        self.ranked_unique_atoms = ranked_unique_atoms
        self.top_5_unique_atoms = ranked_unique_atoms[0:5]
        self.bond_types = ['Single', 'Double', 'Triple', 'Aromatic',
                           'Pi', 'Quadruple', 'Delocalised', 'Universal']

        # The bond-specific message functions.
        self.message_func = nn.ModuleDict()

        for bond in self.bond_types:
            self.message_func[bond] = nn.Sequential(
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.message_size, self.message_size, bias=False),
            )

        # The atom-specific GRU update function for the non-universal and universal nodes.
        self.update_func = nn.ModuleDict()
        self.update_func_universal = nn.ModuleDict()

        for atom in self.top_5_unique_atoms:
            self.update_func[str(atom)] = nn.GRUCell(self.message_size, self.message_size)
            self.update_func_universal[str(atom)] = nn.GRUCell(self.message_size, self.message_size)

        # The catchall GRU update function for the non-universal and universal nodes.
        self.update_func_catchall = nn.GRUCell(self.message_size, self.message_size)
        self.update_func_catchall_universal = nn.GRUCell(self.message_size, self.message_size)

    def edge_propogation(self, g, h_t, bond_type, bond_number):
        '''
        Propogation of a message through a single bond type. Bond number is the index in which
        that bond is listed in self.bond_types.
        '''
        h_t_bond_type = self.message_func[bond_type](h_t)
        m_bond_type = torch.bmm(g[:, bond_number], h_t_bond_type)
        return m_bond_type

    def message_update(self, prior_message, current_message, atom_list, atomic_number, universal_node=False):
        '''
        prior message = h without batches
        current message = m without batches
        atom_list = the list, in order, of all the atomic numbers
        atomic number = the atomic number corresponding to the atom-specific GRU. If atomic_number = None, catchall
        GRU is used.
        universal_node = whether or not this update is for the universal node.
        '''
        if atomic_number is not None:
            if universal_node == False:
                gru_cell = self.update_func[str(atomic_number)]
            else:
                gru_cell = self.update_func_universal[str(atomic_number)]
            # Select all rows which have our given atomic number.
            prior_message_atom_subset = prior_message.index_select(0, torch.where(atom_list == atomic_number)[0])
            current_message_atom_subset = current_message.index_select(0, torch.where(atom_list == atomic_number)[0])

        else:
            if universal_node == False:
                gru_cell = self.update_func_catchall
            else:
                gru_cell = self.update_func_catchall_universal

            # Select all atoms that will be going through the catchall.
            prior_message_atom_subset = prior_message.index_select(0, torch.where(
                ~torch.isin(atom_list, torch.tensor(self.top_5_unique_atoms).to(device)))[0])
            current_message_atom_subset = current_message.index_select(0, torch.where(
                ~torch.isin(atom_list, torch.tensor(self.top_5_unique_atoms).to(device)))[0])

        # Run through GRU.
        updated_message = gru_cell(prior_message_atom_subset, current_message_atom_subset)
        return updated_message

    def forward(self, g, h):
        batch_size = g.size()[0]

        # Padding the atomic representations to some higher dimension, d = message size.
        h_t = torch.cat([h, torch.zeros(h.size()[0], h.size()[1],
                                        self.message_size - h.size()[2]).type_as(h.data)], 2)

        # Finding the order of atoms for the input.
        atom_numbers = h[:, :, 0].view([-1])

        # Message Passing Loop
        for i in range(self.message_passes):
            # Running the padded atomic information through edge propogation. Note that the universal node
            # is treated separately.
            m_non_universal = sum(
                [self.edge_propogation(g, h_t, bond, i) for i, bond in enumerate(self.bond_types[0:-1])])
            m_universal = self.edge_propogation(g, h_t, 'Universal', len(self.bond_types) - 1)

            h_no_batches = h_t.view([-1, h_t.size()[2]])
            m_no_batches = m_non_universal.view([-1, m_non_universal.size()[2]])
            m_uni_no_batches = m_universal.view([-1, m_non_universal.size()[2]])

            gru_output = torch.empty_like(h_no_batches)
            gru_uni_output = torch.empty_like(h_no_batches)

            # Cleaned up GRU Test
            for atom_type in self.top_5_unique_atoms:
                gru_output[torch.where(atom_numbers == atom_type)[0]] = self.message_update(h_no_batches,
                                                                                                m_no_batches,
                                                                                                atom_numbers,
                                                                                                atom_type,
                                                                                                universal_node=False)

                gru_uni_output[torch.where(atom_numbers == atom_type)[0]] = self.message_update(h_no_batches,
                                                                                                    m_uni_no_batches,
                                                                                                    atom_numbers,
                                                                                                    atom_type,
                                                                                                    universal_node=True)
            gru_output[torch.where(~torch.isin(atom_numbers, torch.tensor(self.top_5_unique_atoms).to(device)) == True)[
                0]] = self.message_update(h_no_batches,
                                          m_no_batches,
                                          atom_numbers,
                                          None,
                                          universal_node=False)

            gru_uni_output[torch.where(~torch.isin(atom_numbers, torch.tensor(self.top_5_unique_atoms).to(device)) == True)[
                0]] = self.message_update(h_no_batches,
                                          m_uni_no_batches,
                                          atom_numbers,
                                          None,
                                          universal_node=True)

            gru_total = gru_output + gru_uni_output
            # Putting the batches back in.
            h_t = gru_total.view([batch_size, h_t.size()[1], h_t.size()[2]])

        return h_t