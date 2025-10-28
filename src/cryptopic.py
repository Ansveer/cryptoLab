import argparse
import json
import os
from PIL import Image
import numpy as np
from metrics import generate_metrics


def keystreamGenLCG(n, key):
    # X_0 - начальное значение (0 <= X_0 < m)
    # m >= 2
    # a - множитель, c - приращение (0 <= a < m) (0 <= c < m)
    X_0 = key
    m = 2**64
    a = 43252341515252341
    c = 13243223452435

    keystream = []
    keystream.append(X_0)
    for i in range(1, n):
        X_n = (a*keystream[i - 1] + c) % m
        keystream.append(X_n)

    for i in range(n):
        keystream[i] = keystream[i] % 255

    return keystream


def xorStream(plaintext, keystream, length):
    arr = []
    for i in range(length):
        arr.append(plaintext[i] ^ keystream[i])

    return arr


def arnoldCatEncrypt(arr, width, height, iterations, i):
    tmp = np.zeros_like(arr)
    for _ in range(iterations):
        for y in range(height):
            for x in range(width):
                nx = (x + y) % width
                ny = (i*x + (i+1)*y) % height
                tmp[ny*width + nx] = arr[y*width + x]
        arr = tmp.copy()
    return arr


def arnoldCatDecrypt(arr, width, height, iterations, i):
    tmp = np.zeros_like(arr)
    for _ in range(iterations):
        for y in range(height):
            for x in range(width):
                nx = ((i+1)*x - y) % width
                ny = (-i*x + y) % height
                tmp[ny*width + nx] = arr[y*width + x]
        arr = tmp.copy()
    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["encrypt", "decrypt"], required=True)
    parser.add_argument('--in', dest="_in", required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--algo', choices=["stream", "perm-mix"], required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--iv')
    # parser.add_argument('--nonce')
    parser.add_argument('--meta')
    args = parser.parse_args()

    if args.mode == "encrypt":
        if args.algo == "stream":
            img = np.asarray(Image.open(f"../imgs/{args._in}"))
            height, width, channels = img.shape

            flat_img = img.flatten()

            iv = args.iv if args.iv else os.urandom(16).hex()
            key = int(str(args.key) + str(int(iv, 16)))
            keystream = keystreamGenLCG(len(flat_img), key)

            encrypted_image = xorStream(flat_img, keystream, len(flat_img))

            image = Image.fromarray(np.array(encrypted_image).reshape((height, width, channels)))
            image.save(f"../imgs/{args.out}")

            meta = {
                "algo": f"{args.algo}",
                "IV": f"{iv}",
            }

            with open(f"../results/{args.meta}" if args.meta else "../results/META.json", "w", encoding="utf-8") as file:
                json.dump(meta, file, ensure_ascii=False)
        elif args.algo == "perm-mix":
            img = np.asarray(Image.open(f"../imgs/{args._in}"))
            height, width, channels = img.shape

            flat_img = img.flatten()

            iterations = 7

            iv = args.iv if args.iv else os.urandom(16).hex()
            key = int(str(args.key) + str(int(iv, 16)))
            keystream = keystreamGenLCG(len(flat_img), key)

            encrypted_arr = img.copy()
            for j in range(channels):
                channel = encrypted_arr[:, :, j]
                encrypted_channel = arnoldCatEncrypt(channel.flatten(), width, height, iterations, key)
                encrypted_arr[:, :, j] = encrypted_channel.reshape((height, width))

            encrypted_image = xorStream(encrypted_arr.flatten(), keystream, len(flat_img))

            image = Image.fromarray(np.array(encrypted_image).reshape((height, width, channels)))
            image.save(f"../imgs/{args.out}")

            meta = {
                "algo": f"{args.algo}",
                "IV": f"{iv}",
            }

            with open(f"../results/{args.meta}" if args.meta else "../results/META.json", "w", encoding="utf-8") as file:
                json.dump(meta, file, ensure_ascii=False)
        generate_metrics(args._in, args.out, args.key, args.algo, "encryption_test")
    elif args.mode == "decrypt":
        if args.algo == "stream":
            if args.iv:
                iv = args.iv
            else:
                with open(f"../results/{args.meta}" if args.meta else "../results/META.json", "r", encoding="utf-8") as file:
                    meta = json.load(file)

                if meta["algo"] != "stream":
                    print("Метод шифрования отличается от метода дешифрования")
                    exit(1)

                iv = meta["IV"]

            img = np.asarray(Image.open(f"../imgs/{args._in}"))
            height, width, channels = img.shape

            flat_img = img.flatten()

            key = int(str(args.key) + str(int(iv, 16)))
            keystream = keystreamGenLCG(len(flat_img), key)

            decrypted_image = xorStream(flat_img, keystream, len(flat_img))

            image = Image.fromarray(np.array(decrypted_image).reshape((height, width, channels)))
            image.save(f"../imgs/{args.out}")
        elif args.algo == "perm-mix":
            if args.iv:
                iv = args.iv
            else:
                with open(f"../results/{args.meta}" if args.meta else "../results/META.json", "r", encoding="utf-8") as file:
                    meta = json.load(file)

                if meta["algo"] != "perm-mix":
                    print("Метод шифрования отличается от метода дешифрования")
                    exit(1)

                iv = meta["IV"]

            img = np.asarray(Image.open(f"../imgs/{args._in}"))
            height, width, channels = img.shape

            flat_img = img.flatten()

            iterations = 7

            key = int(str(args.key) + str(int(iv, 16)))
            keystream = keystreamGenLCG(len(flat_img), key)

            decrypted_arr = xorStream(flat_img, keystream, len(flat_img))

            decrypted_image = np.array(decrypted_arr).reshape((height, width, channels))
            for j in range(channels):
                channel = decrypted_image[:, :, j]
                decrypted_channel = arnoldCatDecrypt(channel.flatten(), width, height, iterations, key)
                decrypted_image[:, :, j] = decrypted_channel.reshape((height, width))

            image = Image.fromarray(np.array(decrypted_image).reshape((height, width, channels)))
            image.save(f"../imgs/{args.out}")
