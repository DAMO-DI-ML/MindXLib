import platform
from setuptools import setup

# Read requirements from requirements.txt
def get_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and setuptools requirement
            if line and not line.startswith("#") and not line.startswith("setuptools"):
                requirements.append(line)
    
    # Add platform-specific requirements
    if platform.system() == "Windows":
        requirements.append("pywin32>=228")  # For Windows-specific functionality
    
    return requirements

# Get platform-specific binary paths
def get_binary_paths():
    binary_paths = []
    if platform.system() == "Windows":
        binary_paths.append('explainers/rules/ruleset/bin/*.exe')
    else:  # Linux and other Unix-like systems
        binary_paths.append('explainers/rules/ruleset/bin/*')
    return binary_paths

setup(
    use_scm_version={
        "write_to": "mindxlib/_version.py",
        "write_to_template": '__version__ = "{version}"',
    },
    install_requires=get_requirements(),
    include_package_data=True,
    package_data={
        'mindxlib': get_binary_paths()
    },
    zip_safe=False,
)