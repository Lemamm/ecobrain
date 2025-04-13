import numpy as np

class EcoEnv:
    def __init__(self, data):
        self.data = data
        self.state = self.build_state(data)

        self.action_labels = [
            "Baisser le chauffage",
            "Prendre le vélo",
            "Réduire la viande",
            "Éteindre appareils inutilisés",
            "Planifier covoiturage",
            "Consommer local"
        ]

        self.action_explanations = [
            "Réduire la température permet une économie directe d'énergie.",
            "Le vélo ne produit aucune émission de CO2.",
            "Réduire la viande diminue les GES liés à l’élevage.",
            "Moins d'appareils en veille = moins de pertes énergétiques.",
            "Le covoiturage réduit les émissions par personne.",
            "Les produits locaux réduisent les coûts logistiques et CO2."
        ]

    def build_state(self, data):
        return np.array([
            data["energy"]["electricity_kWh"] / 1000,
            data["energy"]["gas_kWh"] / 1000,
            data["transport"]["co2_emission_kg"] / 100,
            data["food"]["avg_calories"] / 2500,
            data["food"]["meat_ratio"] / 100,
            data["population"] / 1_000_000,
            data["energy"]["renewable_ratio"],
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            np.random.rand()
        ])

    def step(self, action):
        reward = np.random.uniform(0.1, 1.0)
        self.state += np.random.normal(0, 0.02, size=self.state.shape)
        return self.state, reward