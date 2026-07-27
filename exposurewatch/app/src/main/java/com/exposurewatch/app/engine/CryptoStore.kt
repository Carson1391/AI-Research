package com.exposurewatch.app.engine

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.io.File
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Encrypts local baseline/event files at rest with an AES-256-GCM key held in
 * the Android Keystore (hardware-backed where available). The key never leaves
 * the secure element; files on disk are ciphertext.
 */
object CryptoStore {

    private const val KEY_ALIAS = "exposurewatch_store_key"
    private const val ANDROID_KEYSTORE = "AndroidKeyStore"
    private const val IV_LEN = 12
    private const val TAG_BITS = 128

    private fun key(): SecretKey {
        val ks = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        (ks.getEntry(KEY_ALIAS, null) as? KeyStore.SecretKeyEntry)?.let { return it.secretKey }
        val gen = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, ANDROID_KEYSTORE)
        gen.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build()
        )
        return gen.generateKey()
    }

    fun writeEncrypted(file: File, plaintext: String) {
        runCatching {
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.ENCRYPT_MODE, key())
            val iv = cipher.iv
            val ct = cipher.doFinal(plaintext.toByteArray(Charsets.UTF_8))
            file.outputStream().use { out ->
                out.write(iv)
                out.write(ct)
            }
        }
    }

    fun readDecrypted(file: File): String? {
        if (!file.exists()) return null
        return runCatching {
            val all = file.readBytes()
            if (all.size <= IV_LEN) return null
            val iv = all.copyOfRange(0, IV_LEN)
            val ct = all.copyOfRange(IV_LEN, all.size)
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(TAG_BITS, iv))
            String(cipher.doFinal(ct), Charsets.UTF_8)
        }.getOrNull()
    }
}
