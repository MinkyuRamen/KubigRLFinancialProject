# parametric

unknown environment --> parametric assumption : 주가가 OU process를 따른다. (ornstein uhlenbeck process. aka mean-reversion process)

  - parametric assumption 통한 statistical efficiency 확보 (충분한 episode 확보 되었을때)
  - precise dynamics

turbulence_threshold 99% --> 90% 더 보수적으로 수정

(model train) hyperparameters tuned. 

  - optuna
  - A2C convergence issue로 인해 std=1 로 맞춘 hp 사용
