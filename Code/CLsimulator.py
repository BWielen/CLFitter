import numpy as np
import h5py
import constants_values as cv
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import math

##################################################################################
# General functions
##################################################################################

def lorenzian(x, x0, FWHM):
    gamma = FWHM / 2
    return 1/np.pi * gamma / ((x - x0)**2 + gamma**2)

def gaussian(x, x0, FWHM):
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x - x0)**2 / (2 * sigma**2))


###################################################################################
# EELS CL-Edge Functions
###################################################################################
class Edge:
    def __init__(self, file_path, element, edge, energy_values_interp):
        self.file_path = file_path
        self.element = element
        self.edge = edge
        self.energy_values_interp = energy_values_interp
        self.GOS = None
        self.energy_values = None
        self.q_values = None
        self.ionization_edge = None
        self.Delta_E = None
        self.differential_cross_section = None
        self.subshells = None

    
    def read_data(self):
        """
        Reads the GOS data, energy values, q values, and ionization edge.
        """
        edge_path = f'{self.element}/{self.edge}'
        with h5py.File(self.file_path, 'r') as f:
            dataset = f[edge_path]
            self.GOS = dataset['data'][:, :, 0]
            self.energy_values = dataset['free_energies'][:]
            self.q_values = dataset['q'][:]
            self.ionization_edge = dataset['metadata'].attrs['ionization_energy']
            self.Delta_E = self.energy_values + self.ionization_edge
    def apply_chemical_shift(self, shift_range=(-5, 5)):
        """
        Applies a random chemical shift to the ionization edge.
        """
        chemical_shift = np.random.uniform(*shift_range)
        self.Delta_E += chemical_shift


    def calculate_ddcs(self, E_0, beta):
        """
        Calculates the differential cross-section (DDCS).
        """
        self.differential_cross_section = np.zeros(len(self.Delta_E))
        for i, E in enumerate(self.Delta_E):
            Q_min = E**2 / (4 * cv.Rydberg() * cv.T(E_0))
            Q_max = Q_min + 4 * cv.gamma(E_0)**2 * (cv.T(E_0) / cv.Rydberg()) * np.sin(beta/2)**2
            
            log_Q_values = np.linspace(np.log(Q_min), np.log(Q_max), 1000)
            q_interp_values = np.sqrt(np.exp(log_Q_values)) / cv.a_0()
            
            GOS_interp = np.interp(q_interp_values, self.q_values, self.GOS[:, i])
            integral = np.trapezoid(GOS_interp, log_Q_values)
            
            self.differential_cross_section[i] = 4 * np.pi * cv.a_0()**2 * (cv.Rydberg()**2) / (E * cv.T(E_0)) * integral

    def interpolate_ddcs(self):
        """
        Interpolates the DDCS along a predefined axis.
        """
        if self.differential_cross_section is None:
            raise ValueError("DDCS has not been calculated. Run `calculate_ddcs` first.")

        interpolator = interp.interp1d(self.Delta_E, self.differential_cross_section, bounds_error=False, fill_value=0)
        self.differential_cross_section = interpolator(self.energy_values_interp)

    def create_edge_shape(self, E_0, beta):
        """
        Generates the edge shape by calculating DDCS.
        """
        self.read_data()
        # self.apply_chemical_shift()
        self.calculate_ddcs(E_0, beta)
        self.interpolate_ddcs()
        return self.differential_cross_section, self.ionization_edge

###################################################################################
# EELS Low-loss simualtion Class
###################################################################################
class LowLossEELS:
    def __init__(self, E_0, beta, dispersion):
        self.E_0 = E_0
        self.beta = beta
        self.energy_values = np.arange(-50, 50)
        self.low_loss_spectrum = np.zeros_like(self.energy_values, dtype=float)

    def lorenzian(self, x, x0, FWHM):
        gamma = FWHM / 2
        return 1/np.pi * gamma / ((x - x0)**2 + gamma**2)

    def P_n(self, scattering_parameter, n):
        return 1 / math.factorial(n) * scattering_parameter**n * np.exp(-scattering_parameter)

    def elastic_cross_section(self, Z):
        return 1.87e-24 * Z**(4/3) * (cv.v(self.E_0) / cv.c())**-2

    def generate_zlp(self):
        zlp_width = np.random.uniform(1, 3)
        zlp = self.lorenzian(self.energy_values, 0, zlp_width)
        sigma_e = self.elastic_cross_section(26)  # Default Z=26 for example
        scattering_parameter = np.random.uniform(0.1, 1)
        self.low_loss_spectrum += zlp * sigma_e * self.P_n(scattering_parameter, 0)

    def calculate_plasmon_peaks(self, num_peaks=3):
        plasmon_width = np.random.uniform(3, 20)
        plasmon_energy = np.random.uniform(3, 20)
        plasmon_spectrum = (self.energy_values * plasmon_width * plasmon_energy**2) / \
                           ((self.energy_values**2 - plasmon_energy**2)**2 + (plasmon_energy * plasmon_width)**2)
        plasmon_spectrum[self.energy_values < 0] = 0

        Delta_E = self.energy_values[1] - self.energy_values[0]
        shift_idx = plasmon_energy / Delta_E
        scattering_parameter = np.random.uniform(0.1, 1)

        for n in range(1, num_peaks + 1):
            shifted_spectrum = np.roll(plasmon_spectrum, int(n * shift_idx))
            plasmon_spectrum += shifted_spectrum * self.P_n(scattering_parameter, n)
            plasmon_spectrum[self.energy_values < 0] = 0

        plasmon_spectrum /= np.pi * cv.a_0() * cv.m_e() * cv.v(self.E_0)**2 * 1e28 / cv.eVtoJ()
        plasmon_spectrum *= np.log(1 + self.beta**2 / cv.theta_E(plasmon_energy, self.E_0)**2)

        self.low_loss_spectrum += plasmon_spectrum

    def get_spectrum(self):
        self.generate_zlp()
        self.calculate_plasmon_peaks()
        return self.low_loss_spectrum
    
    def convolve_low_loss_spectrum(self, core_loss_spectrum):
        self.get_spectrum()
        
        plural_scattering_spectrum = np.convolve(core_loss_spectrum, self.low_loss_spectrum, mode = 'same')

        return plural_scattering_spectrum




###################################################################################
# EELS CL-Edge Functions
###################################################################################
class SimulateEELSSpectrum:
    def __init__(self, file_path, element, shell, E_0, beta, seed):
        self.file_path = file_path

        self.element = element
        self.shell = shell

        self.E_0 = E_0
        self.beta = beta
        self.edge_coordinates = []

        self.dispersion = None
        self.energy_axis = None
        self.spectrum_values = None

        self.set_seed(seed)
        np.random.seed(seed)
        
    def create_axes(self, start, stop, dispersion):
        self.dispersion = dispersion
        self.energy_axis = np.arange(start, stop, dispersion)
        self.spectrum_values = np.zeros_like(self.energy_axis)
        return self.energy_axis
    
    def set_seed(self, seed):
        np.random.seed(seed)
        
    @staticmethod
    def get_shell_info(shell):
        subshell = {'K': '1',
                    'L': '123',
                    'M': '12345',
                    'N': '1234567'}
        return subshell[shell]

    def calculate_all_edges(self):
        subshells = self.get_shell_info(self.shell)
        for subshell in subshells:
            edge = self.shell + subshell
            edge_object = Edge(self.file_path, self.element, edge, self.energy_axis)
            edge_values, ionization_energy = edge_object.create_edge_shape(self.E_0, self.beta) 
            self.spectrum_values += edge_values
            self.edge_coordinates.append([np.argmax(edge_values), ionization_energy])
            print([max(edge_values), ionization_energy])
        return self.spectrum_values, ionization_energy

    def apply_low_loss(self):
        low_loss_object = LowLossEELS(self.E_0, self.beta, self.dispersion)
        self.spectrum_values = low_loss_object.convolve_low_loss_spectrum(self.spectrum_values)

        return self.spectrum_values

    def add_powerlaw_background(self):
        A = 10**np.random.randint(3,8)
        r = np.random.uniform(2,4)
        random_edge_coordinate = self.edge_coordinates[np.random.randint(len(self.edge_coordinates))]

        jump_reference = A*(random_edge_coordinate[1]/self.energy_axis[0])**-r

        jump_limits = [0.2, 1.5]
        jump = np.random.uniform(*jump_limits)
        self.spectrum_values *= jump*jump_reference/self.spectrum_values[random_edge_coordinate[0]]
        self.spectrum_values += A*(self.energy_axis/self.energy_axis[0])**-r

        return self.spectrum_values

    def add_poissonian_noise(self):
        self.spectrum_values += np.random.normal(0,np.sqrt(self.spectrum_values))
        return self.spectrum_values

    def instrumental_shift(self):
        self.energy_axis += np.random.randint(-5/self.dispersion, 5/self.dispersion) * self.dispersion
        
        return self.spectrum_values
    
    def generate_full_spectrum(self):

        if self.energy_axis is None:
            raise ValueError('The axes are not generated yet. First run "create_axes" and specify the start, stop and dispersion of the spectrum')
        
        self.calculate_all_edges()
        self.apply_low_loss()
        self.add_powerlaw_background()
        # self.add_poissonian_noise()
        self.instrumental_shift()
        return self.spectrum_values
    
    def plot_spectrum(self):
        plt.plot(self.energy_axis, self.spectrum_values)
        plt.xlabel('Energy Loss (eV)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(f'EELS Spectrum for {self.element} {self.shell} shell')
        plt.grid()
        plt.show()
