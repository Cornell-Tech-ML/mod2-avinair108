[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py

## Results

### Simple:

![simpleTensor](https://github.com/user-attachments/assets/1f9f7d3e-4225-4b85-8fed-1b2466977197)

- **Dataset**: Simple
- **Learning Rate**: 0.1
- **Epochs**: 500
- **Hidden Units**: 5
- **Time per epoch**: 0.258s

<img width="400" alt="simplelogtensor" src="https://github.com/user-attachments/assets/7db62276-8136-4272-9be8-668f3a35c8f2">

---

### Diag:

![diagtensor](https://github.com/user-attachments/assets/d0787673-4efb-4648-ba44-897a21aaf889)

- **Dataset**: Diag
- **Learning Rate**: 0.5
- **Epochs**: 500
- **Hidden Units**: 6
- **Time per epoch**: 0.326s

<img width="392" alt="diag_log_tensor" src="https://github.com/user-attachments/assets/a174adc0-f5d6-482f-b61e-ca12f7e7c7e8">

---

### Split:

![splittensor](https://github.com/user-attachments/assets/808f52ef-dbe5-4826-8ffe-abdd667055f3)

- **Dataset**: Split
- **Learning Rate**: 0.1
- **Epochs**: 500
- **Hidden Units**: 10
- **Time per epoch**: 0.735s

<img width="417" alt="splittensor" src="https://github.com/user-attachments/assets/1de6266d-ea41-4d1e-8da4-d5fcb4b4239a">

---

### XOR:

![xortensor](https://github.com/user-attachments/assets/5627da26-f409-4bc1-aa24-22aee91c7078)

- **Dataset**: XOR
- **Learning Rate**: 0.1
- **Epochs**: 500
- **Hidden Units**: 12
- **Time per epoch**: 0.961s

<img width="419" alt="Screenshot 2024-10-22 at 6 21 41 PM" src="https://github.com/user-attachments/assets/dff996a3-5db3-4e38-9d68-74bd327d0922">