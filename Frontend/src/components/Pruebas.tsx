 

import { useState, useRef } from 'react'
import { Camera, Upload, Image as ImageIcon, Zap, CheckCircle, AlertCircle, Loader, XCircle, AlertTriangle } from 'lucide-react'

// Tipos para la respuesta del backend
interface Metricas {
  total_pixeles: number
  pixeles_con_grietas: number
  porcentaje_grietas: number
  num_grietas_detectadas: number
  area_promedio_grieta: number
  area_max_grieta: number
  severidad: string
  estado: string
  confianza: number
}

interface PredictResponse {
  success: boolean
  metricas: Metricas
  result_image: string
  timestamp: string
  error?: string
}

const Pruebas = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // URL del backend
  const API_URL = '/api'

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      // Validar tamaño (máximo 20MB)
      if (file.size > 20 * 1024 * 1024) {
        setError('El archivo es demasiado grande. Máximo 20MB.')
        return
      }

      // Validar tipo
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

      console.log('Enviando a:', `${API_URL}/predict`)

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      console.log('Response status:', response.status)

      if (!response.ok) {
        const contentType = response.headers.get('content-type')
        if (contentType && contentType.includes('application/json')) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Error en la predicción')
        } else {
          const text = await response.text()
          console.error('Response HTML:', text)
          throw new Error('El servidor no respondió correctamente. Verifica la configuración.')
        }
      }

      const data: PredictResponse = await response.json()
      console.log('Resultado:', data)
      setResult(data)

      // Cargar imagen procesada
      if (data.result_image) {
        setProcessedImage(`${API_URL}${data.result_image}`)
      }

    } catch (err) {
      console.error('Error en análisis:', err)
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
      case 'alta': return 'text-red-600'
      case 'media': return 'text-yellow-600'
      case 'baja': return 'text-green-600'
      default: return 'text-gray-600'
    }
  }

  const getSeveridadBg = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta': return 'bg-red-50 border-red-200'
      case 'media': return 'bg-yellow-50 border-yellow-200'
      case 'baja': return 'bg-green-50 border-green-200'
      default: return 'bg-gray-50 border-gray-200'
    }
  }

  return (
    <div className="pt-20">
      <section className="bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50 py-16 min-h-screen">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-5xl font-bold text-gray-800 mb-4 text-center">
            🧪 Prueba el Sistema
          </h2>
          <p className="text-xl text-gray-600 text-center max-w-3xl mx-auto mb-12">
            Analiza imágenes con el modelo de Deep Learning SegFormer B5
          </p>

          {/* Mensaje de Error Global */}
          {error && (
            <div className="max-w-2xl mx-auto mb-6 bg-red-50 border-2 border-red-200 rounded-xl p-4 flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="font-semibold text-red-800">Error</p>
                <p className="text-sm text-red-700">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-red-600 hover:text-red-800"
              >
                <XCircle className="w-5 h-5" />
              </button>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Panel de Captura/Carga */}
            <div className="bg-white rounded-xl shadow-2xl p-8">
              <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <Camera className="w-7 h-7 text-blue-600" />
                Captura de Imagen
              </h3>

              {!selectedImage ? (
                <div className="space-y-4">
                  <button
                    onClick={simulateCameraCapture}
                    className="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white py-5 px-6 rounded-xl font-semibold hover:from-blue-600 hover:to-blue-700 transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl hover:scale-105"
                  >
                    <Camera className="w-6 h-6" />
                    Simular Captura con Raspberry Pi + Arducam
                  </button>

                  <div className="flex items-center gap-4">
                    <div className="flex-1 h-px bg-gray-300"></div>
                    <span className="text-gray-500 font-medium">o</span>
                    <div className="flex-1 h-px bg-gray-300"></div>
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
                    className="w-full bg-gradient-to-r from-purple-500 to-purple-600 text-white py-5 px-6 rounded-xl font-semibold hover:from-purple-600 hover:to-purple-700 transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl hover:scale-105"
                  >
                    <Upload className="w-6 h-6" />
                    Subir Imagen desde tu Dispositivo
                  </button>

                  <div className="mt-8 bg-blue-50 border-2 border-blue-200 rounded-xl p-6">
                    <h4 className="font-semibold text-blue-800 mb-3 flex items-center gap-2 text-lg">
                      <ImageIcon className="w-6 h-6" />
                      Instrucciones:
                    </h4>
                    <ul className="text-sm text-blue-700 space-y-2 ml-6 list-disc">
                      <li>Captura o sube una imagen de una superficie de concreto</li>
                      <li>El sistema analizará automáticamente la presencia de grietas</li>
                      <li>Recibirás un reporte con el nivel de severidad detectado</li>
                      <li>Formatos soportados: PNG, JPG, BMP, TIFF (máximo 20MB)</li>
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Imagen Original */}
                  <div className="rounded-xl overflow-hidden shadow-xl">
                    <img
                      src={selectedImage}
                      alt="Imagen original"
                      className="w-full h-80 object-contain bg-gray-100"
                    />
                    <div className="bg-gray-100 p-2 text-center">
                      <p className="text-sm text-gray-600 font-medium">Imagen Original</p>
                    </div>
                  </div>

                  {/* Botones de acción */}
                  <div className="flex gap-3">
                    {!isProcessing && !result && (
                      <button
                        onClick={analyzeImage}
                        disabled={!selectedFile}
                        className="flex-1 bg-gradient-to-r from-green-500 to-green-600 text-white py-4 px-6 rounded-xl font-semibold hover:from-green-600 hover:to-green-700 transition-all duration-300 flex items-center justify-center gap-2 shadow-lg hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <Zap className="w-5 h-5" />
                        Analizar con IA
                      </button>
                    )}
                    <button
                      onClick={resetTest}
                      className="flex-1 bg-gray-200 text-gray-700 py-4 px-6 rounded-xl font-semibold hover:bg-gray-300 transition-all duration-300 flex items-center justify-center gap-2"
                    >
                      <XCircle className="w-5 h-5" />
                      Nueva Prueba
                    </button>
                  </div>

                  {/* Estado de procesamiento */}
                  {isProcessing && (
                    <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-4 flex items-center gap-3">
                      <Loader className="w-8 h-8 text-blue-600 animate-spin" />
                      <div>
                        <p className="font-semibold text-blue-800 text-lg">Procesando imagen...</p>
                        <p className="text-sm text-blue-600">Aplicando modelo SegFormer B5</p>
                      </div>
                    </div>
                  )}

                  {/* Imagen procesada */}
                  {processedImage && result && result.success && (
                    <div className="rounded-xl overflow-hidden shadow-xl">
                      <img
                        src={processedImage}
                        alt="Imagen procesada"
                        className="w-full h-80 object-contain bg-gray-900"
                        onError={(e) => {
                          console.error('Error cargando imagen procesada')
                          e.currentTarget.style.display = 'none'
                        }}
                      />
                      <div className="bg-gray-100 p-2 text-center">
                        <p className="text-sm text-gray-600 font-medium">Imagen Procesada (Detección de Grietas)</p>
                      </div>
                    </div>
                  )}

                  {/* Resultado */}
                  {result && result.success && (
                    <div className="bg-white rounded-xl border-2 border-gray-200 p-6">
                      {result.metricas.porcentaje_grietas > 0 ? (
                        <>
                          <div className="flex items-center gap-3 mb-4">
                            <AlertCircle className="w-10 h-10 text-yellow-500" />
                            <div>
                              <h4 className="text-xl font-bold text-gray-800">
                                ¡Grietas Detectadas!
                              </h4>
                              <p className={`text-lg font-semibold ${getSeveridadColor(result.metricas.severidad)}`}>
                                Severidad: {result.metricas.severidad}
                              </p>
                            </div>
                          </div>

                          <div className="grid grid-cols-2 gap-3 mb-4">
                            <div className="bg-gray-50 rounded-lg p-3">
                              <p className="text-xs text-gray-600">Grietas detectadas</p>
                              <p className="text-2xl font-bold text-gray-800">{result.metricas.num_grietas_detectadas}</p>
                            </div>
                            <div className="bg-gray-50 rounded-lg p-3">
                              <p className="text-xs text-gray-600">Cobertura</p>
                              <p className="text-2xl font-bold text-gray-800">{result.metricas.porcentaje_grietas.toFixed(2)}%</p>
                            </div>
                            <div className="bg-gray-50 rounded-lg p-3">
                              <p className="text-xs text-gray-600">Área máxima</p>
                              <p className="text-2xl font-bold text-gray-800">{result.metricas.area_max_grieta.toFixed(0)} px</p>
                            </div>
                            <div className="bg-gray-50 rounded-lg p-3">
                              <p className="text-xs text-gray-600">Confianza</p>
                              <p className="text-2xl font-bold text-gray-800">{result.metricas.confianza}%</p>
                            </div>
                          </div>

                          <div className={`border-2 rounded-lg p-4 ${getSeveridadBg(result.metricas.severidad)}`}>
                            <p className="font-medium text-center">
                              {result.metricas.severidad === 'Alta' 
                                ? '⚠️ Se recomienda inspección urgente'
                                : result.metricas.severidad === 'Media'
                                ? '⚠️ Se recomienda inspección profesional'
                                : '✓ Monitoreo continuo recomendado'}
                            </p>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="flex items-center gap-3 mb-4">
                            <CheckCircle className="w-10 h-10 text-green-500" />
                            <div>
                              <h4 className="text-xl font-bold text-gray-800">
                                Sin Grietas Detectadas
                              </h4>
                              <p className="text-gray-600">Estructura en buen estado</p>
                            </div>
                          </div>
                          <div className="bg-green-50 border-2 border-green-200 rounded-lg p-4">
                            <p className="text-green-800 text-center font-medium">
                              ✓ Confianza: {result.metricas.confianza}% - Monitoreo continuo recomendado
                            </p>
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Panel de Información (mantener igual) */}
            <div className="space-y-6">
              {/* ... (mantener el código del panel de información igual) */}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Pruebas