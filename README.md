# nimma-ai-retail-shipment-predictor

Shipment date and time predictor

## Requirements

Poetry \
Python 3.10

## Inference

1. Download and setup

```shell
git clone https://github.com/ommug/nimma-ai-retail-shipment-predictor.git
cd nimma-ai-retail-shipment-predictor
poetry update
```

2. Flask service

```shell
poetry shell
python shipment_wrapper.py
```

3. Postman

###

Endpoint: http://localhost:5010/predict

### Input payload

```
[
    {
        "ITEM_ID": "175-NP7964",
        "ORDERED_QTY": 10,
        "SHIP_MODE": "PARCEL",
        "SCAC_2": "UPS",
        "CARRIER_SERVICE_CODE_1": " ",
        "CARRIER_SERVICE_CODE_2": "60",
        "ZIP_CODE": "93291-9639",
        "ZIP_CODE_1": "79763",
        "ORDER_DATE": "2024-01-25 06:00:00.0",
        "ACTUAL_SHIPMENT_DATE": "2024-01-26 01:49:25.0",
        "SELLER_ORGANIZATION_CODE_1": "PDC",
        "ITEM_WEIGHT": 0.1,
        "SHIPNODE_KEY": "VS",
        "SHIPNODE_KEY_1": "VS"
    }
]
```

### Output payload

```
[
    "output": {
        "Predicted Shipment time": "0.0 days, 20.0 hours, 51.9149169921875 mins. "
    }
]
```
