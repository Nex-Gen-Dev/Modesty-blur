plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// CI passes these: -PgithubRepo=owner/repo -PversionCode=<run number>
val ghRepo = (project.findProperty("githubRepo") as String?) ?: "OWNER/REPO"
val buildNumber = (project.findProperty("versionCode") as String?)?.toIntOrNull() ?: 1

android {
    namespace = "app.modestyblur"
    compileSdk = 34

    defaultConfig {
        applicationId = "app.modestyblur"
        minSdk = 26
        targetSdk = 34
        versionCode = buildNumber
        versionName = "0.1.$buildNumber"
        ndk { abiFilters += listOf("arm64-v8a") }
        buildConfigField("String", "GITHUB_REPO", "\"$ghRepo\"")
        buildConfigField("int", "BUILD_NUMBER", "$buildNumber")
    }

    // Prototype key, committed on purpose so every CI build is signed identically and
    // Android accepts each new APK as an update. Replace with secrets before real distribution.
    signingConfigs {
        create("proto") {
            storeFile = file("prototype-release.p12")
            storePassword = "modestyblur"
            keyAlias = "modestyblur"
            keyPassword = "modestyblur"
            storeType = "pkcs12"
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("proto")
            isMinifyEnabled = false
        }
    }

    buildFeatures { buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += listOf("onnx", "json") }
    lint {
        checkReleaseBuilds = false
        abortOnError = false
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
}
