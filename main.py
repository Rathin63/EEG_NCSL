# %% Section 1
import numpy as np
import matplotlib.pyplot as plt

# Plot sinc function (sin(x)/x)
x1 = np.linspace(-10, 10, 1000)
# Avoid division by zero at x=0
y1 = np.sinc(x1/np.pi)  # numpy's sinc is normalized: sinc(x) = sin(πx)/(πx)

plt.figure(figsize=(10, 6))
plt.plot(x1, y1, 'b-', linewidth=2)
plt.title('sin(x)/x', fontsize=14)
plt.xlabel('x')
plt.ylabel('sin(x)/x')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

# %% Section 1
import numpy as np
import matplotlib.pyplot as plt

# Plot sinc function (sin(x)/x)
x1 = np.linspace(-10, 10, 1000)
# Avoid division by zero at x=0
y1 = np.sin(x1/np.pi)  # numpy's sinc is normalized: sinc(x) = sin(πx)/(πx)

plt.figure(figsize=(10, 6))
plt.plot(x1, y1, 'g--', linewidth=4)
plt.title('sin(x)/x', fontsize=14)
plt.xlabel('x')
plt.ylabel('sin(x)/x')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

