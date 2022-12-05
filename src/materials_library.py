import numpy as np


class Material(object):
    def __init__(self, E, rho, sigma_ult, sigma_yield,
                 name, source,
                 **kwargs):
        super(Material, self).__init__()

        self.name = name
        self.__source = source
        self.E = E
        self.rho = rho
        self.sigma_ult = sigma_ult
        self.sigma_yield = sigma_yield

        self.__SF_yield = np.nan
        self.__SF_ult = np.nan
        self.sigma_max = np.nan

    def __str__(self):
        return self.name

    def set_sigma_crit(self, SF_yield, SF_ult):
        self.__SF_yield = SF_yield
        self.__SF_ult = SF_ult
        self.sigma_max = np.nanmin([self.sigma_ult / self.__SF_ult, self.sigma_yield / self.__SF_yield])


Ti6Al4V = Material(name="Titanium Ti-6Al-4V (Grade 5), Annealed",
                   E=113.0 * 10 ** 9, rho=4480, sigma_yield=0.880 * 10 ** 9, sigma_ult=0.950 * 10 ** 9,
                   source="https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=mtp641")

A7075T6 = Material(name="Aluminum 7075-T6",
                   E=71.7 * 10 ** 9, rho=2810, sigma_yield=0.503 * 10 ** 9, sigma_ult=0.572 * 10 ** 9,
                   source="https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma7075t6")

SST301FH = Material(name="Stainless Steel Type 301, Full Hard, stress relived",
                   E=196.0 * 10 ** 9, rho=8030, sigma_yield=0.965 * 10 ** 9, sigma_ult=1.276 * 10 ** 9,
                   source="https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=NL3014")

SST304 = Material(name="Type 304 Stainless Steel",
                   E=193.0 * 10 ** 9, rho=8000, sigma_yield=0.215 * 10 ** 9, sigma_ult=0.505 * 10 ** 9,
                   source="https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=mq304a")


materials_list = [Ti6Al4V, A7075T6, SST301FH, SST304]