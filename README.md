# PyNILM - Framework for Non-Intrusive Load Monitoring in Python
---

PyNILM is a framework designed to facilitate research in the field of non-intrusive load monitoring (NILM) using the Python programming language. This project provides a structure for the advanced preparation and analysis of data, particularly geared towards machine learning applications.

## Project Structure 🏗️

PyNILM is structured as a Dev Container, providing a consistent environment for the development and execution of experiments. To use this framework, Docker must be installed on the machine, including the Windows Subsystem for Linux (WSL) for Windows users.

## Built on the Shoulders of Giants 🦄

PyNILM leverages the capabilities of [NILMTK](https://github.com/nilmtk/nilmtk), a powerful NILM toolkit, for part of its data processing tasks.

## Examples and Demonstrations 📚

The `notebooks` folder contains practical examples illustrating how to use the package and its functions. These notebooks serve as introductory and practical guides for researchers looking to explore the capabilities of PyNILM.

## Quick Start with VSCode Dev Container 🚀

To facilitate the use of PyNILM, we recommend Visual Studio Code (VSCode) with Dev Containers support. Follow the steps below to set up and start the Dev Container:

1. **Install Docker:** Ensure that Docker is installed on your machine. You can find installation instructions [here](https://docs.docker.com/get-docker/).

2. **Install VSCode:** Download and install Visual Studio Code [here](https://code.visualstudio.com/).

3. **Install Remote - Containers Extension:** In VSCode, install the "Remote - Containers" extension to facilitate Dev Container execution.

4. **Open the Project in VSCode:** Open VSCode and navigate to the PyNILM project directory.

5. **Build the Dev Container:** Upon opening the project, VSCode will automatically detect the Dev Container configuration. Click the green button in the lower right (or press `Ctrl + Shift + P` and choose "Remote-Containers: Reopen in Container") to start the Dev Container build.

6. **Access the Dev Container Environment:** After the build, you will be inside the Dev Container environment, ready to use PyNILM.

## Contributions 🤝

We welcome contributions to enhance and improve PyNILM. To contribute, follow these quick steps:

1. **Fork the Repository:** Click the "Fork" button on the top right of this repository to create your copy.

2. **Clone Your Fork:** Clone your forked repository to your local machine using the command:
```bash
git clone https://github.com/your-username/PyNILM.git
```

3. **Create a Branch:** Create a new branch for your changes:
```bash
git checkout -b feature-name
```

4. **Make Changes:** Make your desired changes and additions to the code.

5. **Commit Changes:** Commit your changes with a descriptive commit message:
```bash
git commit -m "Description of your changes"
```

6. **Push Changes:** Push your changes to your fork on GitHub:
```bash
git push origin feature-name
```

7. **Create a Pull Request:** Open a pull request on the main repository to propose your changes.

## License 📝
This project is distributed under the MIT License. Feel free to use it as needed.

**Enjoy exploring PyNILM!**
---

*Note: Be aware that PyNILM is a work in progress. Evaluate data carefully for application in your research.*