import cv2
import numpy as np
import os

def mask_petri_dish_folder(input_root, output_root, background='black', extensions={'.jpg', '.jpeg', '.png'}):
    os.makedirs(output_root, exist_ok=True)
    total_images = 0
    masked_images = 0
    unmasked_images = 0

    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                input_path = os.path.join(dirpath, filename)

                rel_path = os.path.relpath(dirpath, input_root)
                output_dir = os.path.join(output_root, rel_path)
                os.makedirs(output_dir, exist_ok=True)

                output_filename = os.path.splitext(filename)[0] + ('.png' if background == 'transparent' else '.jpeg')
                output_path = os.path.join(output_dir, output_filename)

                img = cv2.imread(input_path)
                if img is None:
                    print(f"Skipped unreadable file: {input_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)

                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=gray.shape[0]//4,
                    param1=50,
                    param2=25,
                    minRadius=100,
                    maxRadius=300
                )
                total_images += 1

                if circles is not None:
                    masked_images += 1

                    x, y, r = np.uint16(np.around(circles[0][0]))
                    mask = np.zeros_like(gray)
                    shrink_factor = 0.95
                    cv2.circle(mask, (x, y), int(r*shrink_factor), 255, -1)

                    if background == 'transparent':
                        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                        img_bgra[:, :, 3] = mask
                        cv2.imwrite(output_path, img_bgra)
                    else:
                        masked = cv2.bitwise_and(img, img, mask=mask)
                        bg_color = 0 if background == 'black' else 255
                        bg = np.full_like(img, bg_color)
                        inverse_mask = cv2.bitwise_not(mask)
                        bg = cv2.bitwise_and(bg, bg, mask=inverse_mask)
                        final = cv2.add(masked, bg)
                        cv2.imwrite(output_path, final)

                    print(f"Saved: {output_path}")
                else:
                    unmasked_images += 1
                    print(f"Circle not found in: {input_path}")
    print("\n--- Summary ---")
    print(f"Total images processed: {total_images}")
    print(f"Masked images      : {masked_images}")
    print(f" Not masked (no circle): {unmasked_images}")


mask_petri_dish_folder(
    input_root=r"G:\My Drive\MasterThesis\ResizedDataFixed\IBT_32341",
    output_root=r"G:\My Drive\MasterThesis\Circular\IBT_32341",
    background="black" 
)
