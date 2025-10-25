import { useState, useRef } from 'react'
import { Camera, Upload, Image as ImageIcon, Zap, CheckCircle, AlertCircle, Loader, XCircle, AlertTriangle, Info, Settings } from 'lucide-react'

interface Metricas {
  total_pixeles: number
  pixeles_con_grietas: number
  porcentaje_grietas: number
  num_grietas_detectadas: number
  area_promedio_grieta: number
  area_max_grieta: number
  area_min_grieta: number
  severidad: string
  estado: string
  color_severidad: string
  confianza: number
  tta_usado: boolean
}

interface PredictResponse {
  success: boolean
  metricas: Metricas
  result_image?: string
  mask_image?: string
  grietas_image?: string
  imagen_overlay?: string
  imagen_mascara?: string
  imagen_grietas_solas?: string
  timestamp: string
  procesamiento?: {
    tta_usado: boolean
    threshold: number
    target_size: number
  }
  error?: string
}

const Pruebas = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [useTTA, setUseTTA] = useState(true)
  const [showSettings, setShowSettings] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Detectar entorno y configurar URL correcta
  const API_URL = import.meta.env.VITE_API_URL || 
                  (window.location.hostname === 'localhost' 
                    ? 'http://localhost:5001/api' 
                    : '/api')

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (file.size > 20 * 1024 * 1024) {
        setError('El archivo es demasiado grande. Máximo 20MB.')
        return
      }

      const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff']
      if (!validTypes.includes(file.type)) {
        setError('Formato no válido. Use PNG, JPG, BMP o TIFF.')
        return
      }

      setError(null)
      setSelectedFile(file)
      
      const reader = new FileReader()
      reader.onloadend = () => {
        setSelectedImage(reader.result as string)
        setResult(null)
        setProcessedImage(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const simulateCameraCapture = async () => {
    try {
      const response = await fetch('https://images.unsplash.com/photo-1541888946425-d81bb19240f5?w=800&h=600&fit=crop')
      const blob = await response.blob()
      const file = new File([blob], 'raspberry_capture.jpg', { type: 'image/jpeg' })
      
      setSelectedFile(file)
      setSelectedImage(URL.createObjectURL(blob))
      setResult(null)
      setError(null)
      setProcessedImage(null)
    } catch (err) {
      setError('Error al simular captura de cámara')
    }
  }

  const analyzeImage = async () => {
    if (!selectedFile) {
      setError('No hay imagen seleccionada')
      return
    }

    setIsProcessing(true)
    setError(null)
    setResult(null)
    setProcessedImage(null)

    try {
      const formData = new FormData()
      formData.append('image', selectedFile)
      formData.append('use_tta', useTTA.toString())
      formData.append('return_base64', 'true')

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const contentType = response.headers.get('content-type')
        if (contentType && contentType.includes('application/json')) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Error en la predicción')
        } else {
          throw new Error('El servidor no respondió correctamente. Verifica la configuración.')
        }
      }

      const data: PredictResponse = await response.json()
      
      if (!data.success) {
        throw new Error(data.error || 'Error en la predicción')
      }

      setResult(data)

      // Manejar imagen procesada (base64 o URL)
      if (data.imagen_overlay) {
        setProcessedImage(data.imagen_overlay)
      } else if (data.result_image) {
        setProcessedImage(`${API_URL}${data.result_image}`)
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido al analizar la imagen')
    } finally {
      setIsProcessing(false)
    }
  }

  const resetTest = () => {
    setSelectedImage(null)
    setSelectedFile(null)
    setResult(null)
    setError(null)
    setIsProcessing(false)
    setProcessedImage(null)
  }

  const getSeveridadColor = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta': return 'text-red-400'
      case 'media': return 'text-yellow-400'
      case 'baja': return 'text-green-400'
      default: return 'text-slate-400'
    }
  }

  const getSeveridadBg = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta': return 'bg-red-500/10 border-red-500/30'
      case 'media': return 'bg-yellow-500/10 border-yellow-500/30'
      case 'baja': return 'bg-green-500/10 border-green-500/30'
      default: return 'bg-slate-500/10 border-slate-500/30'
    }
  }

  const getSeveridadIcon = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta': return '🔴'
      case 'media': return '🟡'
      case 'baja': return '🟢'
      default: return '⚪'
    }
  }

  return (
    <div className="pt-16 bg-slate-950 min-h-screen">
      <section className="relative py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full px-5 py-2 mb-6">
              <Camera className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">PRUEBAS EN VIVO</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">
              Prueba el Sistema
            </h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Analiza imágenes con el modelo de Deep Learning SegFormer B5 + TTA
            </p>
            
            {/* Toggle TTA */}
            <div className="mt-8 inline-flex items-center gap-4 bg-slate-800/50 border border-slate-700 rounded-full px-6 py-3">
              <Settings className="w-5 h-5 text-slate-400" />
              <span className="text-slate-300 font-medium">Test-Time Augmentation</span>
              <button
                onClick={() => setUseTTA(!useTTA)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  useTTA ? 'bg-cyan-500' : 'bg-slate-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    useTTA ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
              <span className={`text-sm font-semibold ${useTTA ? 'text-cyan-400' : 'text-slate-500'}`}>
                {useTTA ? 'ACTIVADO' : 'DESACTIVADO'}
              </span>
            </div>
            
            {useTTA && (
              <p className="mt-3 text-sm text-cyan-400">
                ⚡ Mayor precisión (4x transformaciones) - Procesamiento más lento
              </p>
            )}
          </div>

          {error && (
            <div className="max-w-3xl mx-auto mb-8">
              <div className="relative group">
                <div className="absolute inset-0 bg-red-500/20 rounded-2xl blur-xl"></div>
                <div className="relative bg-slate-800 border-2 border-red-500/50 rounded-2xl p-4 flex items-start gap-3">
                  <AlertTriangle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="font-semibold text-red-400 mb-1">Error</p>
                    <p className="text-sm text-slate-300">{error}</p>
                  </div>
                  <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300 transition-colors">
                    <XCircle className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Panel de Captura */}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 rounded-3xl blur-2xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8 hover:border-cyan-500/50 transition-all duration-300">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-2 rounded-xl">
                    <Camera className="w-6 h-6 text-white" />
                  </div>
                  Captura de Imagen
                </h3>

                {!selectedImage ? (
                  <div className="space-y-4">
                    <button
                      onClick={simulateCameraCapture}
                      className="group/btn relative w-full overflow-hidden"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-2xl blur-xl opacity-75 group-hover/btn:opacity-100 transition duration-300"></div>
                      <div className="relative bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-5 px-6 rounded-2xl font-semibold transition-all duration-300 flex items-center justify-center gap-3 shadow-lg shadow-cyan-500/50 hover:scale-105">
                        <Camera className="w-6 h-6" />
                        Simular Captura con Raspberry Pi
                      </div>
                    </button>

                    <div className="flex items-center gap-4">
                      <div className="flex-1 h-px bg-slate-700"></div>
                      <span className="text-slate-500 font-medium">o</span>
                      <div className="flex-1 h-px bg-slate-700"></div>
                    </div>

                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleImageUpload}
                      accept="image/png,image/jpeg,image/jpg,image/bmp,image/tiff"
                      className="hidden"
                    />
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="group/btn relative w-full overflow-hidden"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl blur-xl opacity-75 group-hover/btn:opacity-100 transition duration-300"></div>
                      <div className="relative bg-gradient-to-r from-blue-500 to-indigo-600 text-white py-5 px-6 rounded-2xl font-semibold transition-all duration-300 flex items-center justify-center gap-3 shadow-lg shadow-blue-500/50 hover:scale-105">
                        <Upload className="w-6 h-6" />
                        Subir Imagen desde Dispositivo
                      </div>
                    </button>

                    <div className="mt-8 bg-cyan-500/10 border border-cyan-500/30 rounded-2xl p-6">
                      <h4 className="font-semibold text-cyan-400 mb-4 flex items-center gap-2 text-lg">
                        <Info className="w-6 h-6" />
                        Instrucciones
                      </h4>
                      <ul className="text-sm text-slate-300 space-y-3">
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Captura o sube una imagen de una superficie de concreto</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>El sistema analizará automáticamente con IA la presencia de grietas</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Recibirás métricas detalladas y nivel de severidad detectado</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Formatos: PNG, JPG, BMP, TIFF (máx. 20MB)</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="rounded-2xl overflow-hidden border border-slate-700">
                      <img
                        src={selectedImage}
                        alt="Imagen original"
                        className="w-full h-80 object-contain bg-slate-900"
                      />
                      <div className="bg-slate-800 p-3 text-center border-t border-slate-700">
                        <p className="text-sm text-slate-400 font-medium">Imagen Original</p>
                      </div>
                    </div>

                    <div className="flex gap-3">
                      {!isProcessing && !result && (
                        <button
                          onClick={analyzeImage}
                          disabled={!selectedFile}
                          className="flex-1 group/btn relative overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <div className="absolute inset-0 bg-gradient-to-r from-green-600 to-emerald-600 rounded-xl blur-xl opacity-75 group-hover/btn:opacity-100 transition duration-300"></div>
                          <div className="relative bg-gradient-to-r from-green-500 to-emerald-600 text-white py-4 px-6 rounded-xl font-semibold transition-all duration-300 flex items-center justify-center gap-2 shadow-lg shadow-green-500/50 hover:scale-105">
                            <Zap className="w-5 h-5" />
                            Analizar con IA {useTTA && '+ TTA'}
                          </div>
                        </button>
                      )}
                      <button
                        onClick={resetTest}
                        className="flex-1 bg-slate-700 border border-slate-600 text-slate-300 py-4 px-6 rounded-xl font-semibold hover:bg-slate-600 hover:border-slate-500 transition-all duration-300 flex items-center justify-center gap-2"
                      >
                        <XCircle className="w-5 h-5" />
                        Nueva Prueba
                      </button>
                    </div>

                    {isProcessing && (
                      <div className="bg-blue-500/10 border border-blue-500/30 rounded-2xl p-6">
                        <div className="flex items-center gap-4 mb-4">
                          <Loader className="w-10 h-10 text-blue-400 animate-spin" />
                          <div>
                            <p className="font-bold text-blue-400 text-xl">Procesando imagen...</p>
                            <p className="text-sm text-slate-400">
                              {useTTA ? 'Aplicando TTA (4x transformaciones)' : 'Aplicando modelo SegFormer B5'}
                            </p>
                          </div>
                        </div>
                        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                          <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 animate-pulse rounded-full w-full"></div>
                        </div>
                      </div>
                    )}

                    {processedImage && result && result.success && (
                      <div className="rounded-2xl overflow-hidden border border-slate-700">
                        <img
                          src={processedImage}
                          alt="Imagen procesada"
                          className="w-full h-80 object-contain bg-slate-900"
                          onError={(e) => {
                            console.error('Error cargando imagen procesada')
                            e.currentTarget.style.display = 'none'
                          }}
                        />
                        <div className="bg-slate-800 p-3 text-center border-t border-slate-700">
                          <p className="text-sm text-slate-400 font-medium">
                            Imagen Procesada (Detección de Grietas) 
                            {result.metricas.tta_usado && <span className="text-cyan-400"> • TTA Aplicado</span>}
                          </p>
                        </div>
                      </div>
                    )}

                    {result && result.success && (
                      <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6">
                        {result.metricas.porcentaje_grietas > 0 ? (
                          <div className="space-y-4">
                            <div className="flex items-center gap-3 mb-4">
                              <div className="text-4xl">{getSeveridadIcon(result.metricas.severidad)}</div>
                              <div className="flex-1">
                                <h4 className="text-2xl font-bold text-white">
                                  {result.metricas.estado}
                                </h4>
                                <p className={`text-lg font-semibold ${getSeveridadColor(result.metricas.severidad)}`}>
                                  Severidad: {result.metricas.severidad}
                                </p>
                              </div>
                              <AlertCircle className={`w-10 h-10 ${getSeveridadColor(result.metricas.severidad)}`} />
                            </div>

                            <div className="grid grid-cols-2 gap-3">
                              {[
                                { label: 'Grietas detectadas', value: result.metricas.num_grietas_detectadas, icon: '🔍' },
                                { label: 'Cobertura', value: `${result.metricas.porcentaje_grietas.toFixed(2)}%`, icon: '📊' },
                                { label: 'Área máxima', value: `${result.metricas.area_max_grieta.toFixed(0)} px`, icon: '📏' },
                                { label: 'Confianza', value: `${result.metricas.confianza}%`, icon: '✓' },
                              ].map((item, idx) => (
                                <div key={idx} className="bg-slate-900 border border-slate-700 rounded-xl p-4 hover:border-cyan-500/50 transition-all">
                                  <p className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                                    <span>{item.icon}</span>
                                    {item.label}
                                  </p>
                                  <p className="text-2xl font-bold text-white">{item.value}</p>
                                </div>
                              ))}
                            </div>

                            <div className="grid grid-cols-2 gap-3">
                              <div className="bg-slate-900 border border-slate-700 rounded-xl p-3">
                                <p className="text-xs text-slate-500 mb-1">Área promedio</p>
                                <p className="text-lg font-bold text-white">{result.metricas.area_promedio_grieta.toFixed(0)} px</p>
                              </div>
                              <div className="bg-slate-900 border border-slate-700 rounded-xl p-3">
                                <p className="text-xs text-slate-500 mb-1">Total píxeles</p>
                                <p className="text-lg font-bold text-white">{result.metricas.total_pixeles.toLocaleString()}</p>
                              </div>
                            </div>

                            <div className={`border rounded-xl p-4 ${getSeveridadBg(result.metricas.severidad)}`}>
                              <p className={`font-medium text-center ${getSeveridadColor(result.metricas.severidad)}`}>
                                {result.metricas.severidad === 'Alta' 
                                  ? '⚠️ Se recomienda inspección urgente e intervención inmediata'
                                  : result.metricas.severidad === 'Media'
                                  ? '⚠️ Se recomienda inspección profesional programada'
                                  : '✓ Monitoreo continuo recomendado - Estructura estable'}
                              </p>
                            </div>

                            {result.procesamiento && (
                              <div className="bg-slate-900/50 border border-slate-600 rounded-xl p-3 text-xs text-slate-400">
                                <p>Procesamiento: {result.procesamiento.tta_usado ? 'TTA' : 'Estándar'} • 
                                   Umbral: {result.procesamiento.threshold} • 
                                   Resolución: {result.procesamiento.target_size}x{result.procesamiento.target_size}
                                </p>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="space-y-4">
                            <div className="flex items-center gap-3 mb-4">
                              <CheckCircle className="w-12 h-12 text-green-400" />
                              <div className="flex-1">
                                <h4 className="text-2xl font-bold text-white">
                                  {result.metricas.estado}
                                </h4>
                                <p className="text-slate-400">Estructura en buen estado</p>
                              </div>
                            </div>

                            <div className="grid grid-cols-2 gap-3">
                              <div className="bg-slate-900 border border-slate-700 rounded-xl p-4">
                                <p className="text-xs text-slate-500 mb-1">Confianza</p>
                                <p className="text-2xl font-bold text-green-400">{result.metricas.confianza}%</p>
                              </div>
                              <div className="bg-slate-900 border border-slate-700 rounded-xl p-4">
                                <p className="text-xs text-slate-500 mb-1">Total píxeles</p>
                                <p className="text-2xl font-bold text-white">{result.metricas.total_pixeles.toLocaleString()}</p>
                              </div>
                            </div>

                            <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4">
                              <p className="text-green-400 text-center font-medium">
                                ✓ Sin grietas significativas detectadas - Monitoreo continuo recomendado
                              </p>
                            </div>

                            {result.procesamiento && (
                              <div className="bg-slate-900/50 border border-slate-600 rounded-xl p-3 text-xs text-slate-400">
                                <p>Procesamiento: {result.procesamiento.tta_usado ? 'TTA Activado' : 'Estándar'} • 
                                   Umbral: {result.procesamiento.threshold} • 
                                   Resolución: {result.procesamiento.target_size}x{result.procesamiento.target_size}
                                </p>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Panel de Información */}
            <div className="space-y-6">
              <div className="relative group">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-indigo-600/10 rounded-3xl blur-2xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8 hover:border-blue-500/50 transition-all duration-300">
                  <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                    <div className="bg-gradient-to-br from-blue-500 to-indigo-600 p-2 rounded-xl">
                      <ImageIcon className="w-6 h-6 text-white" />
                    </div>
                    Tecnología IA
                  </h3>
                  
                  <div className="space-y-4">
                    <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-5">
                      <h4 className="font-bold text-cyan-400 mb-3 text-lg">SegFormer B5 + TTA</h4>
                      <p className="text-slate-300 text-sm leading-relaxed mb-3">
                        Arquitectura transformer de última generación con Test-Time Augmentation para máxima precisión en segmentación semántica.
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {['Transformer', 'TTA 4x', 'Alta Precisión', 'Deep Learning'].map((tag, idx) => (
                          <span key={idx} className="bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold px-3 py-1 rounded-full">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-5">
                      <h4 className="font-bold text-blue-400 mb-3 text-lg">Características del Sistema</h4>
                      <ul className="space-y-3">
                        {[
                          { icon: '🎯', text: 'Detección automática pixel-perfect' },
                          { icon: '📊', text: 'Métricas detalladas y análisis completo' },
                          { icon: '⚡', text: 'Procesamiento optimizado con TTA' },
                          { icon: '🔍', text: 'Segmentación precisa de grietas' },
                          { icon: '📈', text: 'Análisis de severidad multinivel' },
                          { icon: '💾', text: 'Trazabilidad completa de resultados' },
                        ].map((item, idx) => (
                          <li key={idx} className="flex items-start gap-3 text-slate-300 text-sm">
                            <span className="text-lg">{item.icon}</span>
                            <span>{item.text}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="bg-gradient-to-br from-indigo-500/10 to-blue-600/10 border border-indigo-500/30 rounded-xl p-5">
                      <h4 className="font-bold text-indigo-400 mb-3 text-lg">Niveles de Severidad</h4>
                      <div className="space-y-3">
                        <div className="flex items-center gap-3 bg-green-500/10 rounded-lg p-2">
                          <div className="w-3 h-3 bg-green-400 rounded-full flex-shrink-0"></div>
                          <span className="text-slate-300 text-sm flex-1">Baja: {'<'}1% de cobertura</span>
                          <span className="text-xs text-green-400 font-semibold">Monitoreo</span>
                        </div>
                        <div className="flex items-center gap-3 bg-yellow-500/10 rounded-lg p-2">
                          <div className="w-3 h-3 bg-yellow-400 rounded-full flex-shrink-0"></div>
                          <span className="text-slate-300 text-sm flex-1">Media: 1-15% de cobertura</span>
                          <span className="text-xs text-yellow-400 font-semibold">Atención</span>
                        </div>
                        <div className="flex items-center gap-3 bg-red-500/10 rounded-lg p-2">
                          <div className="w-3 h-3 bg-red-400 rounded-full flex-shrink-0"></div>
                          <span className="text-slate-300 text-sm flex-1">Alta: {'>'}15% de cobertura</span>
                          <span className="text-xs text-red-400 font-semibold">Urgente</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="relative group">
                <div className="absolute inset-0 bg-gradient-to-br from-green-500/10 to-emerald-600/10 rounded-3xl blur-2xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8 hover:border-green-500/50 transition-all duration-300">
                  <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                    <div className="bg-gradient-to-br from-green-500 to-emerald-600 p-2 rounded-xl">
                      <CheckCircle className="w-6 h-6 text-white" />
                    </div>
                    Ventajas del Sistema
                  </h3>
                  
                  <div className="space-y-3">
                    {[
                      { title: 'Sin Riesgos Humanos', desc: 'Inspección remota sin exponer personal', icon: '🛡️' },
                      { title: 'Detección Temprana', desc: 'Identifica problemas antes de que sean críticos', icon: '⏰' },
                      { title: 'Ahorro de Costos', desc: 'Reduce gastos en equipamiento especializado', icon: '💰' },
                      { title: 'Trazabilidad Digital', desc: 'Registro histórico de todas las inspecciones', icon: '📋' },
                      { title: 'Alta Precisión', desc: 'TTA mejora la detección hasta un 15%', icon: '🎯' },
                      { title: 'Análisis Instantáneo', desc: 'Resultados en tiempo real', icon: '⚡' },
                    ].map((item, idx) => (
                      <div key={idx} className="bg-slate-900/50 border border-slate-700 rounded-xl p-4 hover:border-green-500/50 transition-all duration-300 group/item">
                        <div className="flex items-start gap-3">
                          <span className="text-2xl flex-shrink-0">{item.icon}</span>
                          <div className="flex-1">
                            <h4 className="font-semibold text-white mb-1 group-hover/item:text-green-400 transition-colors">{item.title}</h4>
                            <p className="text-slate-400 text-sm">{item.desc}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {result && result.success && (
                <div className="relative group">
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-pink-600/10 rounded-3xl blur-2xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
                  <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 hover:border-purple-500/50 transition-all duration-300">
                    <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                      <div className="bg-gradient-to-br from-purple-500 to-pink-600 p-2 rounded-lg">
                        <Info className="w-5 h-5 text-white" />
                      </div>
                      Detalles del Análisis
                    </h3>
                    
                    <div className="space-y-3 text-sm">
                      <div className="flex justify-between items-center bg-slate-900/50 border border-slate-700 rounded-lg p-3">
                        <span className="text-slate-400">Timestamp</span>
                        <span className="text-white font-mono text-xs">
                          {new Date(result.timestamp).toLocaleString('es-PE')}
                        </span>
                      </div>
                      
                      {result.procesamiento && (
                        <>
                          <div className="flex justify-between items-center bg-slate-900/50 border border-slate-700 rounded-lg p-3">
                            <span className="text-slate-400">TTA Usado</span>
                            <span className={`font-semibold ${result.procesamiento.tta_usado ? 'text-cyan-400' : 'text-slate-400'}`}>
                              {result.procesamiento.tta_usado ? '✓ Activado' : '✗ Desactivado'}
                            </span>
                          </div>
                          
                          <div className="flex justify-between items-center bg-slate-900/50 border border-slate-700 rounded-lg p-3">
                            <span className="text-slate-400">Threshold</span>
                            <span className="text-white font-semibold">{result.procesamiento.threshold}</span>
                          </div>
                          
                          <div className="flex justify-between items-center bg-slate-900/50 border border-slate-700 rounded-lg p-3">
                            <span className="text-slate-400">Resolución</span>
                            <span className="text-white font-semibold">
                              {result.procesamiento.target_size}x{result.procesamiento.target_size}
                            </span>
                          </div>
                        </>
                      )}
                      
                      <div className="flex justify-between items-center bg-slate-900/50 border border-slate-700 rounded-lg p-3">
                        <span className="text-slate-400">Píxeles Analizados</span>
                        <span className="text-white font-semibold">
                          {result.metricas.total_pixeles.toLocaleString()}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center bg-slate-900/50 border border-slate-700 rounded-lg p-3">
                        <span className="text-slate-400">Píxeles con Grietas</span>
                        <span className="text-white font-semibold">
                          {result.metricas.pixeles_con_grietas.toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Información TTA */}
          <div className="mt-12 max-w-5xl mx-auto">
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-blue-600/10 rounded-3xl blur-2xl"></div>
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-8">
                <div className="flex items-start gap-4">
                  <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-3 rounded-xl flex-shrink-0">
                    <Settings className="w-7 h-7 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold text-white mb-3">
                      ¿Qué es Test-Time Augmentation (TTA)?
                    </h3>
                    <p className="text-slate-300 mb-4 leading-relaxed">
                      TTA es una técnica avanzada que mejora la precisión de las predicciones aplicando múltiples transformaciones 
                      a la imagen (rotación, volteos) y promediando los resultados. Esto reduce el ruido y aumenta la confiabilidad 
                      de la detección de grietas.
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                      {[
                        { label: 'Original', icon: '📸' },
                        { label: 'Flip H', icon: '↔️' },
                        { label: 'Flip V', icon: '↕️' },
                        { label: 'Rot 90°', icon: '🔄' },
                      ].map((item, idx) => (
                        <div key={idx} className="bg-slate-900/50 border border-cyan-500/30 rounded-xl p-3 text-center">
                          <div className="text-2xl mb-1">{item.icon}</div>
                          <p className="text-cyan-400 text-sm font-semibold">{item.label}</p>
                        </div>
                      ))}
                    </div>
                    <div className="mt-4 bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-4">
                      <p className="text-cyan-400 text-sm text-center">
                        <strong>Nota:</strong> TTA multiplica el tiempo de procesamiento por 4, pero mejora significativamente 
                        la precisión en casos complejos o imágenes con poca definición.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Pruebas