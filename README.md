# pyStructural

**pyStructural** is a collection of simple yet powerful functions designed to assist structural engineers in their daily activities. This library aims to streamline common calculations and tasks, making it easier for engineers to focus on design and analysis.

## Features

- **Foundation:** Functions to calculate the vertical pressure due to a uniform pressure with the Boussinesq formulation.
- **Section Properties:** Functions to determine properties like area, moment of inertia, and section modulus for various cross-sections.

## Installation

You can install **pyStructural** using pip (in the future):

```bash
pip install pyStructural
```

## Usage

Here's a quick example of how to use pyStructural:

```python
import pyStructural as ps
import numpy as np

# Example: Calculate reactions for a simply supported beam with a point load
span = 6.0  # in meters
load = 10.0  # in kN

# Soil Vertical stress at 5m deep for a 15kN point load located 1.5m away (horizontal distance) 
sigma_y = ps.foundation.boussinesq_pnt(15, 1.5, 5)

# Soil vertical stress at 12m deep for a 150 tonne structure on a rectangular foundation 12m x 10m
bx, bz = 12, 10
q = 150 * 9.81 / (bx * bz)
sigma_y = ps.foundation.boussinesq_udl_rect_cent(q, bx, bz, 12)

# Soil vertical stress for several depth points
y = np.arange(0, 20.001, 0.01)
sigma_y_array = ps.foundation.boussinesq_udl_rect_cent(q, bx, bz, y)
```

## Documentation

For detailed documentation and examples, please refer to the [pyStructural Documentation](future).

## Contributing

We welcome contributions to improve **pyStructural**! Please check out our [contributing guidelines](future) for more information.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, feel free to contact us at edutenorio@gmail.com.

## Acknowledgments

A special thanks to all the contributors who have helped in developing this project.

---

Feel free to adjust the content as per your specific project details and preferences.