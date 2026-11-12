plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.sairam.photorevive"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.sairam.photorevive"
        minSdk = 24
        targetSdk = 34
        // every GitHub build gets a higher number, so it installs as an update
        versionCode = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
        versionName = "1.${System.getenv("GITHUB_RUN_NUMBER") ?: "0"}"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
    }

    // The build's remembered signing key (see .github/workflows/build.yml); falls back to the debug key locally.
    val reviveKey = file("${System.getProperty("user.home")}/.revive-key/photorevive.jks")
    signingConfigs {
        if (reviveKey.exists()) {
            create("revive") {
                storeFile = reviveKey
                storePassword = "photorevive"
                keyAlias = "photorevive"
                keyPassword = "photorevive"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = if (reviveKey.exists()) signingConfigs.getByName("revive")
            else signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { viewBinding = true }
    // AI model files are already compressed; storing them raw lets the app read them fast
    androidResources { noCompress += listOf("onnx") }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += "**/libc++_shared.so"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    // free, offline AI runtime + image processing
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation("org.opencv:opencv:4.10.0")
}
