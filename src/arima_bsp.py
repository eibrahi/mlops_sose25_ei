'''ARIMA'''

# import pandas as pd
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA

# # Dummy-Daten: Aktienkurse über 50 Tage simulieren
# np.random.seed(42)
# data = np.cumsum(np.random.randn(50)) + 100  # Trend simulieren
# df = pd.Series(data)

# # ARIMA-Modell definieren (p=2, d=1, q=2)
# model = ARIMA(df, order=(2, 1, 2))
# model_fit = model.fit()

# # Prognose für die nächsten 10 Tage
# forecast = model_fit.forecast(steps=10)

# # Ergebnisse plotten
# plt.figure(figsize=(10,5))
# plt.plot(df, label="Historische Daten")
# plt.plot(range(len(df), len(df) + len(forecast)), forecast, label="ARIMA-Prognose", linestyle="dashed")
# plt.legend()
# plt.show()
