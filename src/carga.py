import os
import random
import shutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def split_dataset_advanced(source_dir, target_base, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Versión avanzada con paralelización a nivel de imagen
    """
    # Verificar ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Los ratios deben sumar 1"
    
    # Crear directorios destino
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Clases detectadas: {classes}")
    
    # Crear estructura de directorios
    for split in ['train', 'val', 'test']:
        for class_dir in classes:
            os.makedirs(os.path.join(target_base, split, class_dir), exist_ok=True)
    
    # Procesar cada clase
    total_results = []
    
    for class_dir in classes:
        class_path = os.path.join(source_dir, class_dir)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png'))]
        
        if not images:
            total_results.append((class_dir, 0, 0, 0))
            continue
        
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Asignar imágenes a cada split
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }
        
        # Preparar argumentos para copia paralela
        copy_args = []
        for split_name, split_images in splits.items():
            for img in split_images:
                copy_args.append((
                    os.path.join(class_path, img),
                    os.path.join(target_base, split_name, class_dir, img)
                ))
        
        # Copiar archivos en paralelo
        print(f"Copiando {len(copy_args)} imágenes de clase '{class_dir}'...")
        with Pool(processes=cpu_count()) as pool:
            list(tqdm(
                pool.imap(copy_single_file, copy_args),
                total=len(copy_args),
                desc=f"Copiando {class_dir:15}",
                ncols=100
            ))
        
        total_results.append((class_dir, len(splits['train']), len(splits['val']), len(splits['test'])))
    
    # Mostrar resumen
    print("\n" + "="*50)
    print("RESUMEN DE DIVISIÓN")
    print("="*50)
    total_train, total_val, total_test = 0, 0, 0
    
    for class_dir, n_train, n_val, n_test in total_results:
        print(f"Clase {class_dir}: {n_train} train, {n_val} val, {n_test} test")
        total_train += n_train
        total_val += n_val
        total_test += n_test
    
    print(f"\nTOTAL: {total_train} train, {total_val} val, {total_test} test")
    print(f"TOTAL GENERAL: {total_train + total_val + total_test} imágenes")

def copy_single_file(args):
    """
    Copia un solo archivo - función para paralelizar
    """
    src, dst = args
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"Error copiando {src}: {e}")
        return False

# Usar la función
source_directory = "../dataset_limpio"
target_directory = "../dataset_split"

if __name__ == "__main__":
    if os.path.exists(source_directory):
        # Elegir una versión:
        split_dataset_advanced(source_directory, target_directory)  # Solución 2
    else:
        print(f"No se encuentra el directorio fuente: {source_directory}")