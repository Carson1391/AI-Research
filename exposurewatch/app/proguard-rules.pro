# ExposureWatch release keep rules
-keepclasseswithmembers class * extends android.view.View {
    public <init>(android.content.Context, android.util.AttributeSet);
}
-keep class com.exposurewatch.app.MainActivity { *; }
-keep class com.exposurewatch.app.LocateActivity { *; }
-keep class com.exposurewatch.app.OnboardingActivity { *; }
-keep class com.exposurewatch.app.ExposureWatchService { *; }
-keep class com.exposurewatch.app.ir.IrDetectorActivity { *; }
-keepattributes *Annotation*, InnerClasses, Signature
-dontwarn kotlinx.coroutines.**
-dontwarn org.json.**
