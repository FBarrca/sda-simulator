from __future__ import annotations

import numpy as np

WAREHOUSES = ("W_MADRID", "W_BARCELONA", "W_VALENCIA")

CUSTOMERS = (
    "C_MADRID_CENTRO",
    "C_BARCELONA_PORT",
    "C_VALENCIA",
    "C_CASTELLON",
    "C_BILBAO",
    "C_SEVILLA",
    "C_ZARAGOZA",
    "C_MALAGA",
    "C_MURCIA",
    "C_VALLADOLID",
    "C_A_CORUNA",
    "C_ALICANTE",
)

SKUS = ("AMBIENT_FOOD", "COLD_CHAIN", "ELECTRONICS", "PHARMA")

VEHICLES = (
    "V_MAD_1",
    "V_MAD_2",
    "V_BAR_1",
    "V_BAR_2",
    "V_VAL_1",
    "V_VAL_2",
)

VEHICLE_SPECS = {
    "V_MAD_1": ("W_MADRID", 32),
    "V_MAD_2": ("W_MADRID", 22),
    "V_BAR_1": ("W_BARCELONA", 30),
    "V_BAR_2": ("W_BARCELONA", 18),
    "V_VAL_1": ("W_VALENCIA", 26),
    "V_VAL_2": ("W_VALENCIA", 20),
}

WAREHOUSE_COORDS = {
    "W_MADRID": (-3.7038, 40.4168),
    "W_BARCELONA": (2.1734, 41.3851),
    "W_VALENCIA": (-0.3763, 39.4699),
}

CUSTOMER_COORDS = {
    "C_MADRID_CENTRO": (-3.7038, 40.4168),
    "C_BARCELONA_PORT": (2.1734, 41.35),
    "C_VALENCIA": (-0.3763, 39.4699),
    "C_CASTELLON": (-0.0513, 39.9864),
    "C_BILBAO": (-2.935, 43.263),
    "C_SEVILLA": (-5.9845, 37.3891),
    "C_ZARAGOZA": (-0.8891, 41.6488),
    "C_MALAGA": (-4.4214, 36.7213),
    "C_MURCIA": (-1.1307, 37.9922),
    "C_VALLADOLID": (-4.7245, 41.6523),
    "C_A_CORUNA": (-8.4115, 43.3623),
    "C_ALICANTE": (-0.4907, 38.3452),
}

WAREHOUSE_INDEX = {warehouse: index for index, warehouse in enumerate(WAREHOUSES)}
CUSTOMER_INDEX = {customer: index for index, customer in enumerate(CUSTOMERS)}
VEHICLE_INDEX = {vehicle_id: index for index, vehicle_id in enumerate(VEHICLES)}

DISTANCE_KM = np.asarray(
    [
        [10, 620, 355, 425, 395, 530, 320, 530, 400, 215, 600, 420],
        [620, 10, 350, 285, 605, 1000, 310, 970, 575, 730, 1110, 525],
        [355, 350, 10, 70, 610, 660, 310, 630, 225, 560, 960, 170],
    ],
    dtype=float,
)


def distance_km(warehouse: str, customer: str) -> float:
    return float(DISTANCE_KM[WAREHOUSE_INDEX[warehouse], CUSTOMER_INDEX[customer]])


def nearest_warehouse(customer: str) -> str:
    customer_index = CUSTOMER_INDEX[customer]
    warehouse_index = int(np.argmin(DISTANCE_KM[:, customer_index]))
    return WAREHOUSES[warehouse_index]
