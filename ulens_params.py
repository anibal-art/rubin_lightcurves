import astropy.units as u
from astropy import constants as const
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.constants import c, L_sun, sigma_sb, M_jup, M_earth, G
import numpy as np

# Constants
c = const.c
G = const.G
k = 4 * G / (c ** 2)
tstart_Roman = 2461508.763828608
t0 = tstart_Roman + 20


def event_param(random_seed, data_TRILEGAL, system_type, 
t0_range, custom_system=None):
    # print(f'Generation of parameters: {system_type}')

    np.random.seed(random_seed)

    DL = data_TRILEGAL['D_L']
    DS = data_TRILEGAL['D_S']
    mu_rel = data_TRILEGAL['mu_rel']
    logL = data_TRILEGAL['logL'] # log10 of the luminosity in Lsun from TRILEGAL
    logTe = data_TRILEGAL['logTe']  # log10 of effective temperature in K from TRILEGAL
    orbital_period = 0
    semi_major_axis =  np.random.uniform(0.1,28)  

    if system_type == "Planet_systems":      
        star_mass = np.random.uniform(1,100)
        mass_planet = np.random.uniform(1/300,13)
    
    elif system_type =="Binary_stars":
        star_mass = np.random.uniform(1,50)
        mass_planet = np.random.uniform(1,50)*u.M_sun.to("M_jup")
    
    elif system_type == "BH":
        star_mass = np.random.uniform(1,100) # mass of the BH
        mass_planet = 0

    elif system_type == "FFP":
        star_mass = 0  
        mass_planet = np.random.uniform(3.146351865506143e-05,20)
        
    elif system_type == "custom":
        star_mass = set_value("star_mass")
        mass_planet = set_value("planet_mass")

    else:
        raise ValueError(f"Unknown system_type: {system_type}")

    event_params = microlensing_params(system_type, orbital_period, semi_major_axis, DL, star_mass, 
                                               mass_planet, DS, mu_rel, logTe, logL)

    rho = event_params.rho()       
    tE = event_params.tE()
    piE = event_params.piE()

    if t0_range == None:
        # min and max value of opsim
        tstart_rubin =  2460992.515460024
        tend_rubin = 2464601.5487994165
        t0 = np.random.uniform(tstart_rubin-0.5*tE.value, tstart_rubin+0.5*tE.value)  
    else:
        t0 = np.random.uniform(*t0_range)


    if system_type == "Planets_systems":
        u0 = rho.value*np.random.uniform(-3,3)
    else:    
        u0 = np.random.uniform(-1,1)
        
    alpha = np.random.uniform(0,np.pi)        
    angle = np.random.uniform(0,2*np.pi)    
    piEE = piE*np.cos(angle)
    piEN = piE*np.sin(angle)
    
    params_ulens = {'t0':t0,"u0":u0,"tE":tE.value,
                    "piEN":piEN.value,"piEE":piEE.value, 'radius': float(event_params.source_radius().value), 'mass_star':star_mass, "mass_planet": mass_planet, 'thetaE':event_params.theta_E().value}

    if system_type in ["FFP", "Binary_stars","Planets_systems"]:
        params_ulens['rho'] = rho.value
    
    if system_type in ["Binary_stars",'Planets_systems']:
        s = event_params.s()
        q = event_params.mass_ratio()
        params_ulens['s'] = s.value
        params_ulens['q'] = q.value
        params_ulens['alpha'] = alpha

    return params_ulens

  

class microlensing_params:
    
    def __init__(self, name, orbital_period, semi_major_axis, DL, star_mass, mass_planet, DS, mu_rel, logTe, logL):
        self.name = name
        self.orbital_period = (orbital_period * u.day).to(u.year)
        self.semi_major_axis = semi_major_axis * u.au
        self.DL = DL * u.pc
        self.mass_star = star_mass * u.M_sun
        self.mass_planet = mass_planet * u.M_jup
        # self.method = method
        # self.source_radius = source_radius * u.R_sun
        self.DS = DS * u.pc
        self.mu_rel = mu_rel * (u.mas / u.year)
        self.logTe = logTe
        self.logL = logL

    def mass_ratio(self):
        return (self.mass_planet / self.mass_star).decompose()

    def m_lens(self):
        if np.isnan(self.mass_planet):
            return (self.mass_star).decompose().to(u.M_sun)
        else:
            return (self.mass_star + self.mass_planet).decompose().to(u.M_sun)

    def pi_rel(self):
        if self.DL<self.DS:
            # print(u.au, self.DL, u.au / self.DL)
            return ((1 / self.DL) - (1  / self.DS)) * u.rad

        else:
            raise Exception("Invalid distance combination DL>DS")

    def theta_E(self):
        # Calculate theta_E in radians
        theta_E_rad = np.sqrt(k * self.pi_rel() * self.m_lens())
        
        # Convert radians to milliarcseconds (mas)
        theta_E_mas = theta_E_rad.to(u.mas, equivalencies=u.dimensionless_angles())
        
        return theta_E_mas

    
    def tE(self):
        return (self.theta_E() / self.mu_rel).to(u.day)

    def piE(self):
        return (u.au*self.pi_rel() / self.theta_E()).decompose()

    def source_radius(self):
        logL = self.logL
        logTe = self.logTe
        L_star = 10**(logL)
        Teff = (10**(logTe))*u.K
        top = L_star*L_sun
        sigma = sigma_sb
        bot = 4*np.pi*sigma*Teff**4
        Radius = np.sqrt(top/bot).to('R_sun')
        # print('Radius: ',type(Radius), Radius)
        return Radius
    
    def thetas(self):
        if self.DL<self.DS:
            # print('source_radisu:',self.source_radius(),'  DS:', self.DS)
            # Calculate the angular size of the source in radians
            theta_S_rad = (self.source_radius() / self.DS).decompose()
            
            # Convert radians to milliarcseconds (mas)
            theta_S_mas = theta_S_rad.to(u.mas, equivalencies=u.dimensionless_angles())
            # print('thetaS', theta_S_mas)
            return theta_S_mas
        else:
            raise Exception("Invalid distance combination DL>DS")


    def rho(self):
        return (self.thetas() / self.theta_E()).decompose()

    def s(self):
        if self.DL<self.DS:
            # Calculate the angular separation in radians
            s_rad = (self.semi_major_axis / self.DL).decompose()
            
            # Convert radians to milliarcseconds (mas)
            s_mas = s_rad.to(u.mas, equivalencies = u.dimensionless_angles())
            
            # Divide by the Einstein radius to get the normalized separation
            return s_mas / self.theta_E()
        else:
            raise Exception("Invalid distance combination DL>DS")

            
    def u0(self, criterion = "caustic_proximity"):
        random_factor = np.random.uniform(0,3)
        if criterion == "caustic_proximity":
            return random_factor*self.rho() 
        if criterion == "resonant_region":
            return 1/self.s() - self.s()
        # np.sqrt(1 - self.s() ** 2)

    def piE_comp(self):
        phi =  np.random.uniform(0, np.pi) # np.pi/4
        piEE = self.piE() * np.cos(phi)
        piEN = self.piE() * np.sin(phi)
        return piEE, piEN

    def orbital_motion(self, sz=2, a_s=1):
        r_s = sz / self.s()
        n = 2 * np.pi / self.orbital_period
        denominator = a_s * np.sqrt((-1 + 2 * a_s) * (1 + r_s**2))
        velocity_magnitude = n * denominator
    
        def sample_velocities(magnitude):
            # Extract the value of magnitude (without units)
            magnitude_value = magnitude.value
            
            # Generate random velocities
            gamma = np.random.normal(size=3)
            gamma *= magnitude_value / np.linalg.norm(gamma)
            return gamma
    
        # Sample velocities
        gamma1, gamma2, gamma3 = sample_velocities(velocity_magnitude)
        
        # Assign velocities to components
        v_para = gamma1
        v_perp = gamma2
        v_radial = gamma3
        
        return r_s, a_s, v_para, v_perp, v_radial

# def event_param(random_seed, data_TRILEGAL, data_Genulens, system_type, t0_range = [2460413.013828608,2460413.013828608+365.25*8]):
#     # print(f'Generation of parameters: {system_type}')
#     np.random.seed(random_seed)
#     DL = data_Genulens['D_L']
#     DS = data_Genulens['D_S']
#     mu_rel = data_Genulens['mu_rel']
#     logL = data_TRILEGAL['logL'] # log10 of the luminosity in Lsun from TRILEGAL
#     logTe = data_TRILEGAL['logTe']  # log10 of effective temperature in K from TRILEGAL
#     orbital_period = 0
#     semi_major_axis =  np.random.uniform(0.1,28)  

#     if system_type == "Planets_systems":      
#         star_mass = np.random.uniform(1,100)
#         mass_planet = np.random.uniform(1/300,13)
    
#     elif system_type =="Binary_stars":
#         star_mass = np.random.uniform(1,50)
#         mass_planet = np.random.uniform(1,50)*u.M_sun.to("M_jup")
    
#     elif system_type == "BH":
#         star_mass = np.random.uniform(1,100) # mass of the BH
#         mass_planet = 0

#     elif system_type == "FFP":
#         star_mass = 0  
#         mass_planet = np.random.uniform(1/300,20)

#     else:
#         raise ValueError(f"Unknown system_type: {system_type}")
     
#     event_params = microlensing_params(system_type, orbital_period, semi_major_axis, DL, star_mass, 
#                                                mass_planet, DS, mu_rel, logTe, logL)

#     t0 = np.random.uniform(*t0_range)  
#     rho = event_params.rho()       
#     tE = event_params.tE()
#     piE = event_params.piE()
    
#     if system_type == "Planets_systems":
#         u0 = rho*np.random.uniform(-3,3)
#     else:
#         u0 = rho*np.random.uniform(-2,2)
        
#     alpha = np.random.uniform(0,np.pi)        
#     angle = np.random.uniform(0,2*np.pi)    
#     piEE = piE*np.cos(angle)
#     piEN = piE*np.sin(angle)
    
#     params_ulens = {'t0':t0,"u0":u0.value,"tE":tE.value,
#                     "piEN":piEN.value,"piEE":piEE.value}

#     if system_type in ["FFP", "Binary_stars",'Planets_systems']:
#         params_ulens['rho'] = rho.value
    
#     if system_type in ["Binary_stars",'Planets_systems']:
#         s = event_params.s()
#         q = event_params.mass_ratio()
#         params_ulens['s'] = s.value
#         params_ulens['q'] = q.value
#         params_ulens['alpha'] = alpha

#     return params_ulens


def _finite_positive(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x) & (x >= 0)]

def mu_S_from_df(df, pmra_col="pmracosd", pmdec_col="pmdec"):
    """Compute |μ_S| = sqrt(pmra^2 + pmdec^2) [mas/yr] from TRILEGAL df."""
    mu_a = np.asarray(df[pmra_col], float)
    mu_d = np.asarray(df[pmdec_col], float)
    mask = np.isfinite(mu_a) & np.isfinite(mu_d)
    return np.hypot(mu_a[mask], mu_d[mask])

# ---------- Histogram inverse-CDF fit/sampling ----------
def build_hist_icdf_from_muS(mu_S, bins=512, mu_min=0.0, mu_max=None, smooth_sigma=None):
    """
    Build inverse-CDF from |μ_S| histogram (optionally smoothed).
    Returns (x_grid, cdf).
    """
    x = _finite_positive(mu_S)
    if x.size == 0:
        raise ValueError("No valid |μ_S| samples.")
    if mu_max is None:
        q99 = np.quantile(x, 0.999)
        mu_max = max(q99, x.max())

    hist, edges = np.histogram(x, bins=bins, range=(mu_min, mu_max), density=True)
    centers = 0.5*(edges[:-1] + edges[1:])

    if smooth_sigma is not None and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        hist = gaussian_filter1d(hist, sigma=smooth_sigma, mode='nearest')

    hist = np.clip(hist, 0, None)
    # normalize to PDF and build CDF
    dx = centers[1] - centers[0]
    pdf = hist / np.trapz(hist, centers)
    cdf = np.cumsum(pdf) * dx
    cdf = np.clip(cdf, 0, 1)

    # ensure strictly increasing for interpolation
    eps = 1e-12
    cdf = np.maximum.accumulate(cdf + eps*np.arange(cdf.size))
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

    return centers, cdf

def sample_from_hist_icdf(x_grid, cdf, n_samples, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    u = rng.random(n_samples)
    return np.interp(u, cdf, x_grid)

# ---------- Rayleigh fit/sampling ----------
def fit_rayleigh_from_muS(mu_S):
    """MLE for Rayleigh sigma from |μ_S|: sigma^2 = (1/(2n)) * sum(mu^2)."""
    x = _finite_positive(mu_S)
    if x.size == 0:
        raise ValueError("No valid |μ_S| samples.")
    return float(np.sqrt(np.mean(x**2) / 2.0))

def sample_rayleigh(n_samples, sigma, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    u = np.clip(rng.random(n_samples), 1e-12, 1-1e-12)
    return sigma * np.sqrt(-2.0*np.log(1.0 - u))

# ---------- Unified wrapper ----------
def sample_mu_rel_from_muS(df, n_samples=100000, method="hist",
                           pmra_col="pmracosd", pmdec_col="pmdec",
                           hist_bins=512, hist_smooth_sigma=1.0, rng=None):
    """
    Fit |μ_S| from df and sample μ_rel from that fit.
    method: "hist" (nonparam) or "rayleigh" (parametric).
    Returns (mu_rel_samples, mu_S_empirical).
    """
    mu_S = mu_S_from_df(df, pmra_col=pmra_col, pmdec_col=pmdec_col)

    if method == "hist":
        xg, cdf = build_hist_icdf_from_muS(mu_S, bins=hist_bins, smooth_sigma=hist_smooth_sigma)
        mu_rel_samples = sample_from_hist_icdf(xg, cdf, n_samples, rng=rng)
    elif method == "rayleigh":
        sigma = fit_rayleigh_from_muS(mu_S)
        mu_rel_samples = sample_rayleigh(n_samples, sigma, rng=rng)
    else:
        raise ValueError("method must be 'hist' or 'rayleigh'.")

    return mu_rel_samples, mu_S