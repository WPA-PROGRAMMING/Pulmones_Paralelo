import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

#DATASET_DIR = "../Images"
#OUTPUT_DIR = "../dataset_limpio"
DATASET_DIR = "Images"
OUTPUT_DIR = "dataset_limpio"

TARGET_SIZE = (224, 224)
VALID_EXTS = [".png"]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_image_gray(img):
    """Convierte a escala de grises y normaliza al rango [0,1]."""
    img_gray = img.convert("L")  # modo L = grayscale (1 canal)
    arr = np.asarray(img_gray).astype(np.float32)
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)  # normaliza
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def process_single_image(args):
    """Procesa una sola imagen - función para paralelizar"""
    img_path, output_path, img_name = args
    
    try:
        img = Image.open(img_path)
        img = img.resize(TARGET_SIZE)
        img = normalize_image_gray(img)
        output_file = os.path.splitext(img_name)[0] + ".png"
        output_filepath = os.path.join(output_path, output_file)
        img.save(output_filepath)
        return True, None
    except Exception as e:
        return False, f"Error con {img_name}: {e}"

def process_class(cls):
    """Procesa todas las imágenes de una clase en paralelo"""
    input_path = os.path.join(DATASET_DIR, cls)
    output_path = os.path.join(OUTPUT_DIR, cls)
    ensure_dir(output_path)
    
    images = [f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in VALID_EXTS]
    
    if not images:
        return cls, 0, 0, []
    
    # Preparar argumentos para cada imagen
    image_args = [
        (os.path.join(input_path, img_name), output_path, img_name)
        for img_name in images
    ]
    
    # Procesar imágenes de esta clase en paralelo
    successful = 0
    errors = []
    
    # Usar menos procesos para evitar sobrecarga
    num_processes = min(cpu_count(), len(images))
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, image_args),
            total=len(image_args),
            desc=f"Procesando {cls:15}",
            ncols=100
        ))
    
    # Contar resultados
    for success, error in results:
        if success:
            successful += 1
        else:
            errors.append(error)
    
    return cls, successful, len(images), errors

def process_dataset():
    classes = os.listdir(DATASET_DIR)
    print(f"Clases detectadas: {classes}")
    print(f"Número de CPUs disponibles: {cpu_count()}")
    print(f"Procesando {len(classes)} clases...\n")
    
    total_processed = 0
    total_images = 0
    all_errors = []
    
    # Procesar cada clase secuencialmente (pero imágenes en paralelo)
    for cls in classes:
        cls_name, successful, total, errors = process_class(cls)
        total_processed += successful
        total_images += total
        all_errors.extend(errors)
        
        print(f"Clase {cls_name}: {successful}/{total} imágenes procesadas")
        
        # Mostrar errores de esta clase si los hay
        if errors:
            print(f"  Errores en {cls_name}: {len(errors)}")
            for error in errors[:3]:  # Mostrar solo los primeros 3 errores
                print(f"    - {error}")
            if len(errors) > 3:
                print(f"    ... y {len(errors) - 3} más")
    
    # Mostrar resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Total de imágenes procesadas exitosamente: {total_processed}/{total_images}")
    print(f"Porcentaje de éxito: {(total_processed/total_images)*100:.2f}%")
    
    if all_errors:
        print(f"\nErrores totales: {len(all_errors)}")
        # Guardar errores en un archivo
        error_log = os.path.join(OUTPUT_DIR, "errores_procesamiento.txt")
        with open(error_log, 'w') as f:
            for error in all_errors:
                f.write(error + '\n')
        print(f"Detalles de errores guardados en: {error_log}")
    
    print("\n¡Limpieza y normalización completadas!")

if __name__ == "__main__":
    process_dataset()