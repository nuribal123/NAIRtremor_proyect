def compare_files(file1, file2):
    # Leer los archivos
    with open(file1, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()

    with open(file2, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()

    # Convertir el segundo archivo en un conjunto para comparación rápida
    lines2_set = set(lines2)

    # Encontrar líneas que están en file1 pero no en file2
    missing_lines = [line for line in lines1 if line not in lines2_set]

    # Mostrar los resultados
    print("Líneas en", file1, "que no están en", file2, ":")
    for line in missing_lines:
        print(line.strip())

    return missing_lines


# Ejemplo de uso
file1 = "MoBL_ARMS_module2_4_allmusclesLOCKparaOS.osim"  # Cambia esto por la ruta de tu primer archivo
file2 = "MoBL_ARMS_module2_4_allmusclesLOCKparaOSnosup.osim"  # Cambia esto por la ruta de tu segundo archivo
compare_files(file1, file2)