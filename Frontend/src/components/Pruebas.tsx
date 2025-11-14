 


/*

import { useState, useRef, useEffect } from 'react'
import { Camera, Upload, Image as ImageIcon, Zap, CheckCircle, AlertCircle, Loader, XCircle, AlertTriangle, Info, Settings, Compass, Wifi, Video, WifiOff } from 'lucide-react'

interface AnalisisMorfologico {
  patron_general: string
  descripcion_patron: string
  causa_probable: string
  severidad_ajuste: number
  recomendacion: string
  distribucion_orientaciones: {
    horizontal: number
    vertical: number
    diagonal: number
    irregular: number
  }
  num_grietas_analizadas: number
  grietas_principales: Array<{
    id: number
    longitud_px: number
    area_px: number
    ancho_promedio_px: number
    angulo_grados: number | null
    orientacion: string
    aspect_ratio: number
    bbox: {
      x: number
      y: number
      width: number
      height: number
    }
  }>
}

interface Metricas {
  total_pixeles: number
  pixeles_con_grietas: number
  porcentaje_grietas: number
  num_grietas_detectadas: number
  longitud_total_px?: number
  longitud_promedio_px?: number
  longitud_maxima_px?: number
  ancho_promedio_px?: number
  severidad: string
  estado: string
  confianza: number
  confidence_max?: number
  confidence_mean?: number
  analisis_morfologico?: AnalisisMorfologico | null
}

interface Procesamiento {
  architecture: string
  encoder: string
  tta_usado: boolean
  tta_transforms: number
  threshold: number
  target_size: number
  cpu_optimized: boolean
  cpu_threads: number
  max_resolution: number
  original_dimensions?: {
    width: number
    height: number
  }
  output_format: string
}

interface PredictResponse {
  success: boolean
  metricas: Metricas
  imagen_overlay?: string
  timestamp: string
  procesamiento?: Procesamiento
  error?: string
}

interface RaspberryDevice {
  device_id: string
  type: string
  ip_local: string
  capabilities: string[]
  connected_at: string
  has_photo?: boolean
  last_photo_time?: string
}

interface DevicesResponse {
  devices: RaspberryDevice[]
  total: number
  timestamp: string
}

const Pruebas = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [useTTA, setUseTTA] = useState(true)
  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)

  const [raspberryDevices, setRaspberryDevices] = useState<RaspberryDevice[]>([])
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null)
  const [isLoadingDevices, setIsLoadingDevices] = useState(false)
  const [isCapturingFromRaspi, setIsCapturingFromRaspi] = useState(false)
  const [showRaspberryPanel, setShowRaspberryPanel] = useState(false)
  const [streamUrl, setStreamUrl] = useState<string | null>(null)
  const [isStreamActive, setIsStreamActive] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamImgRef = useRef<HTMLImageElement>(null)

  // ✅ API_URL CORRECTO
  const API_URL = import.meta.env.VITE_API_URL ||
                  (window.location.hostname === 'localhost'
                    ? 'http://localhost:5000/api'
                    : '/api')

  useEffect(() => {
    loadRaspberryDevices()
    const interval = setInterval(loadRaspberryDevices, 10000)
    return () => clearInterval(interval)
  }, [])

  const loadRaspberryDevices = async () => {
    setIsLoadingDevices(true)
    try {
      const response = await fetch(`${API_URL}/devices`)
      if (response.ok) {
        const data: DevicesResponse = await response.json()
        setRaspberryDevices(data.devices)
        console.log('📱 Dispositivos conectados:', data.devices)
      }
    } catch (err) {
      console.error('Error al cargar dispositivos:', err)
    } finally {
      setIsLoadingDevices(false)
    }
  }

  const captureAndAnalyzeFromRaspberry = async (deviceId: string) => {
    setIsCapturingFromRaspi(true)
    setError(null)
    setSelectedDevice(deviceId)

    try {
      console.log(`📸 Solicitando foto a ${deviceId}...`)

      const cmdResponse = await fetch(`${API_URL}/send_command/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'take_photo' })
      })
      if (!cmdResponse.ok) {
        throw new Error('Error al enviar comando al Raspberry Pi')
      }
      console.log('✅ Comando enviado, esperando foto...')

      await new Promise(resolve => setTimeout(resolve, 3000))

      const photoResponse = await fetch(`${API_URL}/photo/${deviceId}`)

      if (!photoResponse.ok) {
        throw new Error('No se pudo obtener la foto del Raspberry Pi')
      }
      const photoData = await photoResponse.json()

      if (!photoData.success || !photoData.image) {
        throw new Error('Foto vacía o inválida')
      }
      console.log('📸 Foto recibida desde Raspberry Pi')

      const base64Data = photoData.image.split(',')[1] || photoData.image
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'image/jpeg' })
      const file = new File([blob], `raspberry_${deviceId}_${Date.now()}.jpg`, { type: 'image/jpeg' })

      setSelectedFile(file)
      setSelectedImage(photoData.image.includes('data:') ? photoData.image : `data:image/jpeg;base64,${photoData.image}`)
      setResult(null)
      setProcessedImage(null)
      console.log('✅ Foto lista para análisis')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al capturar desde Raspberry Pi')
      console.error('❌ Error:', err)
    } finally {
      setIsCapturingFromRaspi(false)
    }
  }

  // ✅ FUNCIÓN CORREGIDA PARA STREAMING
  const getStreamUrl = async (deviceId: string) => {
    try {
      setSelectedDevice(deviceId)

      // ✅ USAR PROXY DEL BACKEND CON TIMESTAMP PARA EVITAR CACHE
      const timestamp = Date.now()
      const url = `${API_URL}/stream/${deviceId}?t=${timestamp}`

      setStreamUrl(url)
      setIsStreamActive(true)

      console.log('📹 Stream URL (proxy HTTPS):', url)
      return url

    } catch (err) {
      console.error('Error al obtener stream URL:', err)
      setError('No se pudo obtener la URL de streaming')
    }
    return null
  }

  const stopStream = () => {
    setStreamUrl(null)
    setIsStreamActive(false)
    setSelectedDevice(null)
  }

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

  const openCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      })

      setStream(mediaStream)
      setIsCameraOpen(true)

      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream
        }
      }, 100)
    } catch (err) {
      console.error('Error al acceder a la cámara:', err)
      setError('No se pudo acceder a la cámara. Verifica los permisos.')
    }
  }

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    if (!context) return
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera_capture_${Date.now()}.jpg`, { type: 'image/jpeg' })
        setSelectedFile(file)
        setSelectedImage(URL.createObjectURL(blob))
        setResult(null)
        setProcessedImage(null)
        closeCamera()
      }
    }, 'image/jpeg', 0.95)
  }

  const closeCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    setIsCameraOpen(false)
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
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const contentType = response.headers.get('content-type')
        if (contentType?.includes('application/json')) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Error en la predicción')
        } else {
          throw new Error(`Error del servidor: ${response.status}`)
        }
      }
      const data: PredictResponse = await response.json()

      if (!data.success) {
        throw new Error(data.error || 'Error en la predicción')
      }
      setResult(data)
      if (data.imagen_overlay) {
        setProcessedImage(data.imagen_overlay)
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
    closeCamera()
    stopStream()
  }

  const getSeveridadColor = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta':
      case 'media-alta':
        return 'text-red-400'
      case 'media':
        return 'text-yellow-400'
      case 'baja':
        return 'text-green-400'
      case 'sin grietas':
        return 'text-slate-400'
      default:
        return 'text-slate-400'
    }
  }

  const getSeveridadBg = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta':
      case 'media-alta':
        return 'bg-red-500/10 border-red-500/30'
      case 'media':
        return 'bg-yellow-500/10 border-yellow-500/30'
      case 'baja':
        return 'bg-green-500/10 border-green-500/30'
      case 'sin grietas':
        return 'bg-slate-500/10 border-slate-500/30'
      default:
        return 'bg-slate-500/10 border-slate-500/30'
    }
  }

  const getSeveridadIcon = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta':
      case 'media-alta':
        return '🔴'
      case 'media':
        return '🟡'
      case 'baja':
        return '🟢'
      case 'sin grietas':
        return '✅'
      default:
        return '⚪'
    }
  }

  const getPatronIcon = (patron: string) => {
    switch (patron) {
      case 'horizontal': return '↔️'
      case 'vertical': return '↕️'
      case 'diagonal_escalera': return '↗️'
      case 'ramificada_mapa': return '🗺️'
      case 'mixto': return '🔀'
      case 'irregular': return '🌀'
      case 'superficial': return '📏'
      case 'sin_grietas': return '✅'
      default: return '❓'
    }
  }

  const getOrientacionColor = (orientacion: string) => {
    switch (orientacion) {
      case 'horizontal': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      case 'vertical': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'diagonal': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
      case 'irregular': return 'bg-purple-500/20 text-purple-400 border-purple-500/30'
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500/30'
    }
  }

  const safeToFixed = (value: number | undefined, decimals: number = 2): string => {
    return value !== undefined && value !== null ? value.toFixed(decimals) : '0.00'
  }

  return (
    <div className="pt-16 bg-slate-950 min-h-screen">
      <section className="relative py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full px-5 py-2 mb-6">
              <Camera className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">PRUEBAS EN VIVO v3.5 + RASPBERRY PI</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">
              Prueba el Sistema
            </h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              UNet++ EfficientNet-B8 + TTA + Análisis Morfológico + Integración Raspberry Pi
            </p>

            <div className="mt-8 flex flex-wrap justify-center gap-4">
              <div className="inline-flex items-center gap-4 bg-slate-800/50 border border-slate-700 rounded-full px-6 py-3">
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
                  {useTTA ? 'ACTIVADO (6x)' : 'DESACTIVADO'}
                </span>
              </div>
              <button
                onClick={() => {
                  setShowRaspberryPanel(!showRaspberryPanel)
                  if (!showRaspberryPanel) loadRaspberryDevices()
                }}
                className="inline-flex items-center gap-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white px-6 py-3 rounded-full font-semibold hover:scale-105 transition-all shadow-lg shadow-purple-500/50"
              >
                {raspberryDevices.length > 0 ? (
                  <>
                    <Wifi className="w-5 h-5" />
                    <span>{raspberryDevices.length} Raspberry Pi Conectados</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-5 h-5" />
                    <span>Buscar Raspberry Pi</span>
                  </>
                )}
              </button>
            </div>
            {showRaspberryPanel && (
              <div className="mt-8 max-w-4xl mx-auto">
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-white flex items-center gap-3">
                      <Wifi className="w-6 h-6 text-purple-400" />
                      Dispositivos Raspberry Pi
                    </h3>
                    <button
                      onClick={loadRaspberryDevices}
                      disabled={isLoadingDevices}
                      className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all disabled:opacity-50"
                    >
                      {isLoadingDevices ? (
                        <Loader className="w-4 h-4 animate-spin" />
                      ) : (
                        '🔄 Actualizar'
                      )}
                    </button>
                  </div>
                  {raspberryDevices.length === 0 ? (
                    <div className="text-center py-8">
                      <WifiOff className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                      <p className="text-slate-400">No hay dispositivos conectados</p>
                      <p className="text-sm text-slate-500 mt-2">
                        Inicia el cliente WebSocket en tu Raspberry Pi
                      </p>
                    </div>
                  ) : (
                    <div className="grid gap-4">
                      {raspberryDevices.map((device) => (
                        <div
                          key={device.device_id}
                          className={`bg-slate-900 border-2 rounded-xl p-5 transition-all ${
                            selectedDevice === device.device_id
                              ? 'border-purple-500 shadow-lg shadow-purple-500/50'
                              : 'border-slate-700 hover:border-slate-600'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-4">
                            <div>
                              <h4 className="text-lg font-bold text-white flex items-center gap-2">
                                <Camera className="w-5 h-5 text-purple-400" />
                                {device.device_id}
                              </h4>
                              <p className="text-sm text-slate-400 mt-1">
                                📍 {device.ip_local} • {device.type}
                              </p>
                              {device.has_photo && (
                                <p className="text-xs text-green-400 mt-1">
                                  📸 Última foto: {device.last_photo_time}
                                </p>
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                              <span className="text-xs text-green-400 font-semibold">ONLINE</span>
                            </div>
                          </div>
                          <div className="flex flex-wrap gap-2 mb-4">
                            {device.capabilities.map((cap, idx) => (
                              <span
                                key={idx}
                                className="bg-purple-500/20 text-purple-400 text-xs font-semibold px-3 py-1 rounded-full border border-purple-500/30"
                              >
                                {cap}
                              </span>
                            ))}
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <button
                              onClick={() => captureAndAnalyzeFromRaspberry(device.device_id)}
                              disabled={isCapturingFromRaspi}
                              className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                            >
                              {isCapturingFromRaspi ? (
                                <Loader className="w-5 h-5 animate-spin" />
                              ) : (
                                <Camera className="w-5 h-5" />
                              )}
                              Capturar Foto
                            </button>
                            {device.capabilities.includes('streaming') && (
                              <button
                                onClick={() => {
                                  if (isStreamActive && streamUrl && selectedDevice === device.device_id) {
                                    stopStream()
                                  } else {
                                    getStreamUrl(device.device_id)
                                  }
                                }}
                                className={`py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all flex items-center justify-center gap-2 ${
                                  isStreamActive && selectedDevice === device.device_id
                                    ? 'bg-red-500 hover:bg-red-600 text-white'
                                    : 'bg-gradient-to-r from-green-500 to-emerald-600 text-white'
                                }`}
                              >
                                {isStreamActive && selectedDevice === device.device_id ? (
                                  <>
                                    <XCircle className="w-5 h-5" />
                                    Detener Stream
                                  </>
                                ) : (
                                  <>
                                    <Video className="w-5 h-5" />
                                    Ver Stream
                                  </>
                                )}
                              </button>
                            )}
                          </div>
                          {/* ✅ STREAMING EMBEBIDO *//*}
                          {isStreamActive && streamUrl && selectedDevice === device.device_id && (
                            <div className="mt-4 bg-black rounded-xl overflow-hidden border border-purple-500/50">
                              <img
                                ref={streamImgRef}
                                src={streamUrl}
                                alt="Streaming en vivo"
                                className="w-full h-auto"
                                onError={(e) => {
                                  console.error('❌ Error al cargar streaming')
                                  setError('No se pudo conectar al streaming. Verifica que el servidor de la Raspberry Pi esté activo.')
                                  stopStream()
                                }}
                                onLoad={() => {
                                  console.log('✅ Streaming cargado correctamente')
                                }}
                              />
                              <div className="bg-slate-900 p-3 text-center">
                                <p className="text-sm text-purple-400 font-semibold flex items-center justify-center gap-2">
                                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                                  STREAMING EN VIVO • 720p @ 25 FPS
                                </p>
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
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
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 rounded-3xl blur-2xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8 hover:border-cyan-500/50 transition-all duration-300">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-2 rounded-xl">
                    <Camera className="w-6 h-6 text-white" />
                  </div>
                  Captura de Imagen
                </h3>
                {isCameraOpen && (
                  <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
                    <div className="relative max-w-4xl w-full">
                      <button
                        onClick={closeCamera}
                        className="absolute top-4 right-4 z-10 bg-red-500 hover:bg-red-600 text-white p-3 rounded-full transition-all"
                      >
                        <XCircle className="w-6 h-6" />
                      </button>

                      <div className="bg-slate-900 rounded-2xl overflow-hidden border border-slate-700">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          className="w-full h-auto"
                        />

                        <div className="p-6 flex justify-center gap-4">
                          <button
                            onClick={capturePhoto}
                            className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white px-8 py-4 rounded-xl font-semibold flex items-center gap-3 hover:scale-105 transition-all shadow-lg shadow-cyan-500/50"
                          >
                            <Camera className="w-6 h-6" />
                            Capturar Foto
                          </button>
                        </div>
                      </div>
                    </div>
                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                )}
                {!selectedImage ? (
                  <div className="space-y-4">
                    <button
                      onClick={openCamera}
                      className="group/btn relative w-full overflow-hidden"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl blur-xl opacity-75 group-hover/btn:opacity-100 transition duration-300"></div>
                      <div className="relative bg-gradient-to-r from-purple-500 to-pink-600 text-white py-5 px-6 rounded-2xl font-semibold transition-all duration-300 flex items-center justify-center gap-3 shadow-lg shadow-purple-500/50 hover:scale-105">
                        <Camera className="w-6 h-6" />
                        Tomar Foto con Cámara
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
                          <span>Captura o sube una imagen de estructura de concreto</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Puedes usar tu Raspberry Pi conectado para capturar fotos y ver streaming en vivo</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>El sistema detecta grietas y analiza su patrón morfológico</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Recibe diagnóstico con causa probable y nivel de severidad</span>
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
                              {useTTA ? 'UNet++ B8 + TTA (6x) + Análisis Morfológico' : 'UNet++ B8 Estándar'}
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
                            setError('No se pudo cargar la imagen procesada')
                          }}
                        />
                        <div className="bg-slate-800 p-3 text-center border-t border-slate-700">
                          <p className="text-sm text-slate-400 font-medium">
                            Resultado Procesado
                            {result.procesamiento?.tta_usado && <span className="text-cyan-400"> • TTA {result.procesamiento.tta_transforms}x</span>}
                            {result.procesamiento?.cpu_optimized && <span className="text-green-400"> • CPU Optimizado</span>}
                          </p>
                        </div>
                      </div>
                    )}
                    {result && result.success && (
                      <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6">
                        {result.metricas.num_grietas_detectadas > 0 && result.metricas.porcentaje_grietas > 0 ? (
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
                            {result.metricas.analisis_morfologico && (
                              <div className="bg-gradient-to-br from-purple-500/10 to-pink-600/10 border border-purple-500/30 rounded-xl p-5 space-y-4">
                                <div className="flex items-center gap-3">
                                  <Compass className="w-6 h-6 text-purple-400" />
                                  <h5 className="font-bold text-purple-400 text-lg">Análisis Morfológico Avanzado</h5>
                                </div>

                                <div className="bg-slate-900/50 border border-purple-500/30 rounded-lg p-4">
                                  <div className="flex items-start gap-3 mb-2">
                                    <span className="text-2xl">{getPatronIcon(result.metricas.analisis_morfologico.patron_general)}</span>
                                    <div className="flex-1">
                                      <p className="font-semibold text-white capitalize text-lg">
                                        Patrón: {result.metricas.analisis_morfologico.patron_general.replace('_', ' ')}
                                      </p>
                                      <p className="text-sm text-slate-300 mt-1">
                                        {result.metricas.analisis_morfologico.descripcion_patron}
                                      </p>
                                    </div>
                                  </div>

                                  <div className="mt-3 bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
                                    <p className="text-sm text-orange-400">
                                      <strong>Causa probable:</strong> {result.metricas.analisis_morfologico.causa_probable}
                                    </p>
                                  </div>
                                  <div className={`mt-3 border rounded-lg p-3 ${getSeveridadBg(result.metricas.severidad)}`}>
                                    <p className={`text-sm font-medium ${getSeveridadColor(result.metricas.severidad)}`}>
                                      <strong>📋 Recomendación:</strong> {result.metricas.analisis_morfologico.recomendacion}
                                    </p>
                                  </div>
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                  {Object.entries(result.metricas.analisis_morfologico.distribucion_orientaciones || {}).map(([tipo, count]) => (
                                    count > 0 && (
                                      <div key={tipo} className={`border rounded-lg p-3 text-center ${getOrientacionColor(tipo)}`}>
                                        <p className="text-xs font-medium capitalize mb-1">{tipo}</p>
                                        <p className="text-2xl font-bold">{count}</p>
                                      </div>
                                    )
                                  ))}
                                </div>
                                {result.metricas.analisis_morfologico.grietas_principales &&
                                 result.metricas.analisis_morfologico.grietas_principales.length > 0 && (
                                  <div className="bg-slate-900/50 border border-slate-600 rounded-lg p-3">
                                    <p className="text-xs text-slate-400 mb-3 font-semibold flex items-center gap-2">
                                      🔍 Top {result.metricas.analisis_morfologico.grietas_principales.length} Grietas Analizadas
                                      <span className="text-cyan-400">
                                        (de {result.metricas.num_grietas_detectadas} totales)
                                      </span>
                                    </p>
                                    <div className="space-y-2">
                                      {result.metricas.analisis_morfologico.grietas_principales.slice(0, 5).map((grieta) => (
                                        <div key={grieta.id} className="flex items-center justify-between text-xs bg-slate-800 rounded p-2">
                                          <div className="flex-1">
                                            <span className="text-slate-300 font-semibold">
                                              #{grieta.id} • {grieta.orientacion}
                                            </span>
                                            {grieta.angulo_grados !== null && (
                                              <span className="text-slate-500 ml-2">
                                                ({safeToFixed(grieta.angulo_grados, 1)}°)
                                              </span>
                                            )}
                                          </div>
                                          <div className="text-right">
                                            <div className="text-cyan-400 font-bold">
                                              {safeToFixed(grieta.longitud_px, 0)}px
                                            </div>
                                            <div className="text-slate-500 text-xs">
                                              {safeToFixed(grieta.area_px, 0)}px²
                                            </div>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                            <div className="grid grid-cols-2 gap-3">
                              {[
                                { label: 'Grietas detectadas', value: result.metricas.num_grietas_detectadas, icon: '🔍' },
                                { label: 'Cobertura', value: `${safeToFixed(result.metricas.porcentaje_grietas)}%`, icon: '📊' },
                                { label: 'Longitud máxima', value: `${safeToFixed(result.metricas.longitud_maxima_px, 0)} px`, icon: '📏' },
                                { label: 'Confianza', value: `${safeToFixed(result.metricas.confianza, 1)}%`, icon: '✓' },
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
                            {result.procesamiento && (
                              <div className="bg-slate-900/50 border border-slate-600 rounded-xl p-3 text-xs text-slate-400 space-y-1">
                                <p>
                                  🏗️ <strong>{result.procesamiento.architecture}</strong> + <strong>{result.procesamiento.encoder}</strong>
                                </p>
                                <p>
                                  ⚡ {result.procesamiento.tta_usado ? `TTA (${result.procesamiento.tta_transforms}x)` : 'Estándar'} •
                                  Umbral: {result.procesamiento.threshold} •
                                  Resolución: {result.procesamiento.target_size}px
                                </p>
                                {result.procesamiento.cpu_optimized && (
                                  <p className="text-green-400">
                                    🚀 CPU Optimizado ({result.procesamiento.cpu_threads} threads) •
                                    Max: {result.procesamiento.max_resolution}px •
                                    Salida: {result.procesamiento.output_format}
                                  </p>
                                )}
                                {result.procesamiento.original_dimensions && (
                                  <p>
                                    📐 Original: {result.procesamiento.original_dimensions.width}x{result.procesamiento.original_dimensions.height}px
                                  </p>
                                )}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="space-y-4">
                            <div className="flex items-center gap-3 mb-4">
                              <CheckCircle className="w-12 h-12 text-green-400" />
                              <div className="flex-1">
                                <h4 className="text-2xl font-bold text-white">{result.metricas.estado}</h4>
                                <p className="text-slate-400">Estructura en excelente estado</p>
                              </div>
                            </div>
                            <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4">
                              <p className="text-green-400 text-center font-medium flex items-center justify-center gap-2">
                                <CheckCircle className="w-5 h-5" />
                                Sin grietas significativas detectadas
                              </p>
                            </div>
                            {result.procesamiento && (
                              <div className="bg-slate-900/50 border border-slate-600 rounded-xl p-3 text-xs text-slate-400">
                                <p>
                                  {result.procesamiento.architecture} + {result.procesamiento.encoder} •
                                  {result.procesamiento.tta_usado ? ` TTA (${result.procesamiento.tta_transforms}x)` : ' Estándar'} •
                                  Confianza: {safeToFixed(result.metricas.confianza, 1)}%
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
            <div className="space-y-6">
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-blue-500 to-indigo-600 p-2 rounded-xl">
                    <ImageIcon className="w-6 h-6 text-white" />
                  </div>
                  Tecnología IA v3.5
                </h3>

                <div className="space-y-4">
                  <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-5">
                    <h4 className="font-bold text-cyan-400 mb-3 text-lg">UNet++ EfficientNet-B8</h4>
                    <p className="text-slate-300 text-sm leading-relaxed mb-3">
                      Arquitectura encoder-decoder con Test-Time Augmentation y análisis morfológico condicional optimizado para CPU.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {['UNet++', 'EfficientNet-B8', 'TTA 6x', 'Morfología', 'CPU Opt', 'Raspberry Pi'].map((tag, idx) => (
                        <span key={idx} className="bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold px-3 py-1 rounded-full">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="bg-gradient-to-br from-indigo-500/10 to-blue-600/10 border border-indigo-500/30 rounded-xl p-5">
                    <h4 className="font-bold text-indigo-400 mb-3 text-lg">Patrones Detectados</h4>
                    <div className="space-y-2">
                      {[
                        { icon: '↔️', label: 'Horizontal', causa: 'Flexión, presión lateral', severidad: 'Media' },
                        { icon: '↕️', label: 'Vertical', causa: 'Cargas pesadas, asentamientos', severidad: 'Alta' },
                        { icon: '↗️', label: 'Diagonal', causa: 'Esfuerzos cortantes', severidad: 'Alta' },
                        { icon: '🗺️', label: 'Ramificada', causa: 'Contracción térmica', severidad: 'Baja' },
                      ].map((item, idx) => (
                        <div key={idx} className="bg-slate-900/50 rounded-lg p-3 flex items-start gap-3 hover:bg-slate-900 transition-all">
                          <span className="text-xl">{item.icon}</span>
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <p className="text-white font-semibold text-sm">{item.label}</p>
                              <span className={`text-xs px-2 py-0.5 rounded-full ${getSeveridadBg(item.severidad)}`}>
                                {item.severidad}
                              </span>
                            </div>
                            <p className="text-slate-400 text-xs">{item.causa}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="bg-gradient-to-br from-green-500/10 to-emerald-600/10 border border-green-500/30 rounded-xl p-5">
                    <h4 className="font-bold text-green-400 mb-3 text-lg flex items-center gap-2">
                      <Zap className="w-5 h-5" />
                      Optimizaciones CPU
                    </h4>
                    <ul className="space-y-2 text-sm text-slate-300">
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Análisis morfológico solo si hay grietas detectadas</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Top 10 grietas más grandes (procesamiento rápido)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Resize inteligente (max 2048px sin pérdida)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Integración con Raspberry Pi vía WebSocket</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Streaming en vivo a través de proxy HTTPS</span>
                      </li>
                    </ul>
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
*/



















/*



import { useState, useRef, useEffect } from 'react'
import { Camera, Upload, Image as ImageIcon, Zap, CheckCircle, AlertCircle, Loader, XCircle, AlertTriangle, Info, Settings, Compass, Wifi, Video, WifiOff, RefreshCw } from 'lucide-react'

// ═══════════════════════════════════════════════════════════════════════════
// INTERFACES (SIN CAMBIOS)
// ═══════════════════════════════════════════════════════════════════════════

interface AnalisisMorfologico {
  patron_general: string
  descripcion_patron: string
  causa_probable: string
  severidad_ajuste: number
  recomendacion: string
  distribucion_orientaciones: {
    horizontal: number
    vertical: number
    diagonal: number
    irregular: number
  }
  num_grietas_analizadas: number
  grietas_principales: Array<{
    id: number
    longitud_px: number
    area_px: number
    ancho_promedio_px: number
    angulo_grados: number | null
    orientacion: string
    aspect_ratio: number
    bbox: {
      x: number
      y: number
      width: number
      height: number
    }
  }>
}

interface Metricas {
  total_pixeles: number
  pixeles_con_grietas: number
  porcentaje_grietas: number
  num_grietas_detectadas: number
  longitud_total_px?: number
  longitud_promedio_px?: number
  longitud_maxima_px?: number
  ancho_promedio_px?: number
  severidad: string
  estado: string
  confianza: number
  confidence_max?: number
  confidence_mean?: number
  analisis_morfologico?: AnalisisMorfologico | null
}

interface Procesamiento {
  architecture: string
  encoder: string
  tta_usado: boolean
  tta_transforms: number
  threshold: number
  target_size: number
  cpu_optimized?: boolean
  cpu_threads?: number
  max_resolution?: number
  original_dimensions?: {
    width: number
    height: number
  }
  output_format?: string
}

interface PredictResponse {
  success: boolean
  metricas: Metricas
  imagen_overlay?: string
  timestamp: string
  procesamiento?: Procesamiento
  error?: string
}

// ✅ INTERFAZ ACTUALIZADA PARA REST API
interface RaspberryDevice {
  device_id: string
  type: string
  ip_local: string
  stream_port: number
  streaming_active: boolean
  stream_url_direct: string
  stream_url_proxy: string
  capabilities: string[]
  connected_at: string
  last_seen_ago: number
  has_photo?: boolean
  last_photo_time?: string
}

interface DevicesResponse {
  devices: RaspberryDevice[]
  total: number
  timestamp: string
}

const Pruebas = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [useTTA, setUseTTA] = useState(true)
  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)

  // ✅ ESTADOS PARA RASPBERRY PI (REST API)
  const [raspberryDevices, setRaspberryDevices] = useState<RaspberryDevice[]>([])
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null)
  const [isLoadingDevices, setIsLoadingDevices] = useState(false)
  const [isCapturingFromRaspi, setIsCapturingFromRaspi] = useState(false)
  const [showRaspberryPanel, setShowRaspberryPanel] = useState(false)
  const [streamUrl, setStreamUrl] = useState<string | null>(null)
  const [isStreamActive, setIsStreamActive] = useState(false)
  const [isStartingStream, setIsStartingStream] = useState(false)
  const [isStoppingStream, setIsStoppingStream] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // ✅ API_URL CORRECTO (funciona con tu Nginx)
  const API_URL = import.meta.env.VITE_API_URL ||
                  (window.location.hostname === 'localhost'
                    ? 'http://localhost:5000'
                    : '')

  // ═══════════════════════════════════════════════════════════════════════════
  // ✅ CARGAR DISPOSITIVOS RASPBERRY PI (REST API)
  // ═══════════════════════════════════════════════════════════════════════════

  useEffect(() => {
    loadRaspberryDevices()
    const interval = setInterval(loadRaspberryDevices, 10000) // Cada 10s
    return () => clearInterval(interval)
  }, [])

  const loadRaspberryDevices = async () => {
    setIsLoadingDevices(true)
    try {
      console.log('📡 Cargando dispositivos desde:', `${API_URL}/api/rpi/devices`)
      
      const response = await fetch(`${API_URL}/api/rpi/devices`)
      
      if (response.ok) {
        const data: DevicesResponse = await response.json()
        setRaspberryDevices(data.devices)
        console.log('✅ Dispositivos conectados:', data.devices)
      } else {
        console.error('❌ Error al cargar dispositivos:', response.status)
      }
    } catch (err) {
      console.error('❌ Error al cargar dispositivos:', err)
    } finally {
      setIsLoadingDevices(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ✅ CAPTURAR FOTO DESDE RASPBERRY PI (REST API)
  // ═══════════════════════════════════════════════════════════════════════════

  const captureAndAnalyzeFromRaspberry = async (deviceId: string) => {
    setIsCapturingFromRaspi(true)
    setError(null)
    setSelectedDevice(deviceId)

    try {
      console.log(`📸 Solicitando captura a ${deviceId}...`)

      // ✅ ENVIAR COMANDO DE CAPTURA (REST API)
      const cmdResponse = await fetch(`${API_URL}/api/rpi/capture/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          resolution: '1920x1080',
          format: 'jpg'
        })
      })

      if (!cmdResponse.ok) {
        throw new Error('Error al enviar comando al Raspberry Pi')
      }

      console.log('✅ Comando enviado, esperando foto...')

      // ✅ ESPERAR A QUE EL RASPBERRY CAPTURE Y SUBA LA FOTO (3-5 segundos)
      await new Promise(resolve => setTimeout(resolve, 5000))

      // ✅ OBTENER ÚLTIMA FOTO DEL RASPBERRY PI
      const photoResponse = await fetch(`${API_URL}/api/rpi/latest-photo/${deviceId}`)

      if (!photoResponse.ok) {
        throw new Error('No se pudo obtener la foto del Raspberry Pi')
      }

      const photoData = await photoResponse.json()

      if (!photoData.success || !photoData.image) {
        throw new Error('Foto vacía o inválida')
      }

      console.log('✅ Foto recibida desde Raspberry Pi')

      // ✅ CONVERTIR BASE64 A FILE
      const base64Data = photoData.image.split(',')[1] || photoData.image
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'image/jpeg' })
      const file = new File([blob], `raspberry_${deviceId}_${Date.now()}.jpg`, { type: 'image/jpeg' })

      setSelectedFile(file)
      setSelectedImage(photoData.image.includes('data:') ? photoData.image : `data:image/jpeg;base64,${photoData.image}`)
      setResult(null)
      setProcessedImage(null)
      
      console.log('✅ Foto lista para análisis')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al capturar desde Raspberry Pi')
      console.error('❌ Error:', err)
    } finally {
      setIsCapturingFromRaspi(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ✅ INICIAR STREAMING DESDE RASPBERRY PI (REST API)
  // ═══════════════════════════════════════════════════════════════════════════

  const startStreaming = async (deviceId: string) => {
    setIsStartingStream(true)
    setError(null)
    setSelectedDevice(deviceId)

    try {
      console.log(`🎬 Iniciando streaming en ${deviceId}...`)

      // ✅ ENVIAR COMANDO DE START STREAMING
      const response = await fetch(`${API_URL}/api/rpi/streaming/start/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) {
        throw new Error('Error al iniciar streaming')
      }

      console.log('✅ Comando de inicio enviado')

      // ✅ ESPERAR A QUE EL RASPBERRY INICIE EL STREAMING
      await new Promise(resolve => setTimeout(resolve, 3000))

      // ✅ USAR PROXY DEL BACKEND PARA EVITAR MIXED CONTENT (HTTPS)
      const proxyUrl = `${API_URL}/api/stream/${deviceId}`
      
      setStreamUrl(proxyUrl)
      setIsStreamActive(true)

      console.log('✅ Streaming activo:', proxyUrl)

      // ✅ RECARGAR DISPOSITIVOS PARA ACTUALIZAR ESTADO
      loadRaspberryDevices()

    } catch (err) {
      console.error('❌ Error al iniciar streaming:', err)
      setError('No se pudo iniciar el streaming')
    } finally {
      setIsStartingStream(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ✅ DETENER STREAMING (REST API)
  // ═══════════════════════════════════════════════════════════════════════════

  const stopStreaming = async (deviceId: string) => {
    setIsStoppingStream(true)
    setError(null)

    try {
      console.log(`🛑 Deteniendo streaming en ${deviceId}...`)

      // ✅ ENVIAR COMANDO DE STOP STREAMING
      const response = await fetch(`${API_URL}/api/rpi/streaming/stop/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) {
        throw new Error('Error al detener streaming')
      }

      console.log('✅ Comando de detención enviado')

      setStreamUrl(null)
      setIsStreamActive(false)
      setSelectedDevice(null)

      // ✅ RECARGAR DISPOSITIVOS PARA ACTUALIZAR ESTADO
      await loadRaspberryDevices()

    } catch (err) {
      console.error('❌ Error al detener streaming:', err)
      setError('No se pudo detener el streaming')
    } finally {
      setIsStoppingStream(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // FUNCIONES DE CÁMARA LOCAL (SIN CAMBIOS)
  // ═══════════════════════════════════════════════════════════════════════════

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (file.size > 25 * 1024 * 1024) {
        setError('El archivo es demasiado grande. Máximo 25MB.')
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

  const openCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      })

      setStream(mediaStream)
      setIsCameraOpen(true)

      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream
        }
      }, 100)
    } catch (err) {
      console.error('Error al acceder a la cámara:', err)
      setError('No se pudo acceder a la cámara. Verifica los permisos.')
    }
  }

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    if (!context) return
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera_capture_${Date.now()}.jpg`, { type: 'image/jpeg' })
        setSelectedFile(file)
        setSelectedImage(URL.createObjectURL(blob))
        setResult(null)
        setProcessedImage(null)
        closeCamera()
      }
    }, 'image/jpeg', 0.95)
  }

  const closeCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    setIsCameraOpen(false)
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
      
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const contentType = response.headers.get('content-type')
        if (contentType?.includes('application/json')) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Error en la predicción')
        } else {
          throw new Error(`Error del servidor: ${response.status}`)
        }
      }
      
      const data: PredictResponse = await response.json()

      if (!data.success) {
        throw new Error(data.error || 'Error en la predicción')
      }
      
      setResult(data)
      if (data.imagen_overlay) {
        setProcessedImage(data.imagen_overlay)
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
    closeCamera()
    setStreamUrl(null)
    setIsStreamActive(false)
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // FUNCIONES DE UTILIDAD (SIN CAMBIOS)
  // ═══════════════════════════════════════════════════════════════════════════

  const getSeveridadColor = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta':
      case 'media-alta':
        return 'text-red-400'
      case 'media':
        return 'text-yellow-400'
      case 'baja':
        return 'text-green-400'
      case 'sin grietas':
        return 'text-slate-400'
      default:
        return 'text-slate-400'
    }
  }

  const getSeveridadBg = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta':
      case 'media-alta':
        return 'bg-red-500/10 border-red-500/30'
      case 'media':
        return 'bg-yellow-500/10 border-yellow-500/30'
      case 'baja':
        return 'bg-green-500/10 border-green-500/30'
      case 'sin grietas':
        return 'bg-slate-500/10 border-slate-500/30'
      default:
        return 'bg-slate-500/10 border-slate-500/30'
    }
  }

  const getSeveridadIcon = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta':
      case 'media-alta':
        return '🔴'
      case 'media':
        return '🟡'
      case 'baja':
        return '🟢'
      case 'sin grietas':
        return '✅'
      default:
        return '⚪'
    }
  }

  const getPatronIcon = (patron: string) => {
    switch (patron) {
      case 'horizontal': return '↔️'
      case 'vertical': return '↕️'
      case 'diagonal_escalera': return '↗️'
      case 'ramificada_mapa': return '🗺️'
      case 'mixto': return '🔀'
      case 'irregular': return '🌀'
      case 'superficial': return '📏'
      case 'sin_grietas': return '✅'
      default: return '❓'
    }
  }

  const getOrientacionColor = (orientacion: string) => {
    switch (orientacion) {
      case 'horizontal': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      case 'vertical': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'diagonal': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
      case 'irregular': return 'bg-purple-500/20 text-purple-400 border-purple-500/30'
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500/30'
    }
  }

  const safeToFixed = (value: number | undefined, decimals: number = 2): string => {
    return value !== undefined && value !== null ? value.toFixed(decimals) : '0.00'
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER (CON ADAPTACIONES PARA REST API)
  // ═══════════════════════════════════════════════════════════════════════════

  return (
    <div className="pt-16 bg-slate-950 min-h-screen">
      <section className="relative py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full px-5 py-2 mb-6">
              <Camera className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">PRUEBAS EN VIVO v4.2 + RASPBERRY PI (REST API)</span>
            </div>
            
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">
              Prueba el Sistema
            </h2>
            
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              UNet++ EfficientNet-B8 + TTA + Análisis Morfológico + Raspberry Pi REST API
            </p>

            {/* ✅ CONTROLES SUPERIORES *//*}
            <div className="mt-8 flex flex-wrap justify-center gap-4">
              {/* TTA Toggle *//*}
              <div className="inline-flex items-center gap-4 bg-slate-800/50 border border-slate-700 rounded-full px-6 py-3">
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
                  {useTTA ? 'ACTIVADO (6x)' : 'DESACTIVADO'}
                </span>
              </div>

              {/* ✅ BOTÓN RASPBERRY PI *//*}
              <button
                onClick={() => {
                  setShowRaspberryPanel(!showRaspberryPanel)
                  if (!showRaspberryPanel) loadRaspberryDevices()
                }}
                className="inline-flex items-center gap-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white px-6 py-3 rounded-full font-semibold hover:scale-105 transition-all shadow-lg shadow-purple-500/50"
              >
                {raspberryDevices.length > 0 ? (
                  <>
                    <Wifi className="w-5 h-5" />
                    <span>{raspberryDevices.length} Raspberry Pi Conectados</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-5 h-5" />
                    <span>Buscar Raspberry Pi</span>
                  </>
                )}
              </button>
            </div>

            {/* ✅ PANEL RASPBERRY PI (REST API) *//*}
            {showRaspberryPanel && (
              <div className="mt-8 max-w-4xl mx-auto">
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-white flex items-center gap-3">
                      <Wifi className="w-6 h-6 text-purple-400" />
                      Dispositivos Raspberry Pi (REST API)
                    </h3>
                    <button
                      onClick={loadRaspberryDevices}
                      disabled={isLoadingDevices}
                      className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all disabled:opacity-50 flex items-center gap-2"
                    >
                      {isLoadingDevices ? (
                        <Loader className="w-4 h-4 animate-spin" />
                      ) : (
                        <RefreshCw className="w-4 h-4" />
                      )}
                      Actualizar
                    </button>
                  </div>

                  {raspberryDevices.length === 0 ? (
                    <div className="text-center py-8">
                      <WifiOff className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                      <p className="text-slate-400">No hay dispositivos conectados</p>
                      <p className="text-sm text-slate-500 mt-2">
                        Inicia el cliente bash en tu Raspberry Pi
                      </p>
                    </div>
                  ) : (
                    <div className="grid gap-4">
                      {raspberryDevices.map((device) => (
                        <div
                          key={device.device_id}
                          className={`bg-slate-900 border-2 rounded-xl p-5 transition-all ${
                            selectedDevice === device.device_id
                              ? 'border-purple-500 shadow-lg shadow-purple-500/50'
                              : 'border-slate-700 hover:border-slate-600'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-4">
                            <div>
                              <h4 className="text-lg font-bold text-white flex items-center gap-2">
                                <Camera className="w-5 h-5 text-purple-400" />
                                {device.device_id}
                              </h4>
                              <p className="text-sm text-slate-400 mt-1">
                                📍 {device.ip_local} • {device.type}
                              </p>
                              {device.has_photo && (
                                <p className="text-xs text-green-400 mt-1">
                                  📸 Última foto: {device.last_photo_time}
                                </p>
                              )}
                              <p className="text-xs text-slate-500 mt-1">
                                ⏱️ Última conexión: hace {device.last_seen_ago}s
                              </p>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                              <span className="text-xs text-green-400 font-semibold">ONLINE</span>
                            </div>
                          </div>

                          {/* Capabilities *//*}
                          <div className="flex flex-wrap gap-2 mb-4">
                            {device.capabilities.map((cap, idx) => (
                              <span
                                key={idx}
                                className="bg-purple-500/20 text-purple-400 text-xs font-semibold px-3 py-1 rounded-full border border-purple-500/30"
                              >
                                {cap}
                              </span>
                            ))}
                          </div>

                          {/* ✅ BOTONES DE ACCIÓN *//*}
                          <div className="grid grid-cols-2 gap-3">
                            {/* Capturar Foto *//*}
                            <button
                              onClick={() => captureAndAnalyzeFromRaspberry(device.device_id)}
                              disabled={isCapturingFromRaspi}
                              className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                            >
                              {isCapturingFromRaspi && selectedDevice === device.device_id ? (
                                <Loader className="w-5 h-5 animate-spin" />
                              ) : (
                                <Camera className="w-5 h-5" />
                              )}
                              Capturar Foto
                            </button>

                            {/* Streaming Toggle *//*}
                            {device.capabilities.includes('streaming') && (
                              <button
                                onClick={() => {
                                  if (device.streaming_active || (isStreamActive && selectedDevice === device.device_id)) {
                                    stopStreaming(device.device_id)
                                  } else {
                                    startStreaming(device.device_id)
                                  }
                                }}
                                disabled={isStartingStream || isStoppingStream}
                                className={`py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all flex items-center justify-center gap-2 disabled:opacity-50 ${
                                  device.streaming_active || (isStreamActive && selectedDevice === device.device_id)
                                    ? 'bg-red-500 hover:bg-red-600 text-white'
                                    : 'bg-gradient-to-r from-green-500 to-emerald-600 text-white'
                                }`}
                              >
                                {(isStartingStream || isStoppingStream) && selectedDevice === device.device_id ? (
                                  <Loader className="w-5 h-5 animate-spin" />
                                ) : device.streaming_active || (isStreamActive && selectedDevice === device.device_id) ? (
                                  <>
                                    <XCircle className="w-5 h-5" />
                                    Detener Stream
                                  </>
                                ) : (
                                  <>
                                    <Video className="w-5 h-5" />
                                    Iniciar Stream
                                  </>
                                )}
                              </button>
                            )}
                          </div>

                          {/* ✅ STREAMING IFRAME (PROXY BACKEND) *//*}
                          {isStreamActive && streamUrl && selectedDevice === device.device_id && (
                            <div className="mt-4 bg-black rounded-xl overflow-hidden border border-purple-500/50">
                              <iframe
                                src={streamUrl}
                                className="w-full aspect-video"
                                allow="camera; microphone"
                                title={`Stream ${device.device_id}`}
                                onError={() => {
                                  console.error('❌ Error al cargar streaming iframe')
                                  setError('No se pudo conectar al streaming')
                                }}
                              />
                              <div className="bg-slate-900 p-3 text-center">
                                <p className="text-sm text-purple-400 font-semibold flex items-center justify-center gap-2">
                                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                                  STREAMING EN VIVO • WebRTC • MediaMTX
                                </p>
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* ✅ ERROR DISPLAY *//*}
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

          {/* ✅ GRID PRINCIPAL (IMAGEN + RESULTADOS) *//*}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* COLUMNA IZQUIERDA - CAPTURA/ANÁLISIS *//*}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 rounded-3xl blur-2xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8 hover:border-cyan-500/50 transition-all duration-300">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-2 rounded-xl">
                    <Camera className="w-6 h-6 text-white" />
                  </div>
                  Captura de Imagen
                </h3>

                {/* ✅ MODAL CÁMARA *//*}
                {isCameraOpen && (
                  <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
                    <div className="relative max-w-4xl w-full">
                      <button
                        onClick={closeCamera}
                        className="absolute top-4 right-4 z-10 bg-red-500 hover:bg-red-600 text-white p-3 rounded-full transition-all"
                      >
                        <XCircle className="w-6 h-6" />
                      </button>

                      <div className="bg-slate-900 rounded-2xl overflow-hidden border border-slate-700">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          className="w-full h-auto"
                        />

                        <div className="p-6 flex justify-center gap-4">
                          <button
                            onClick={capturePhoto}
                            className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white px-8 py-4 rounded-xl font-semibold flex items-center gap-3 hover:scale-105 transition-all shadow-lg shadow-cyan-500/50"
                          >
                            <Camera className="w-6 h-6" />
                            Capturar Foto
                          </button>
                        </div>
                      </div>
                    </div>
                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                )}

                {!selectedImage ? (
                  <div className="space-y-4">
                    {/* Botón Cámara *//*}
                    <button
                      onClick={openCamera}
                      className="group/btn relative w-full overflow-hidden"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl blur-xl opacity-75 group-hover/btn:opacity-100 transition duration-300"></div>
                      <div className="relative bg-gradient-to-r from-purple-500 to-pink-600 text-white py-5 px-6 rounded-2xl font-semibold transition-all duration-300 flex items-center justify-center gap-3 shadow-lg shadow-purple-500/50 hover:scale-105">
                        <Camera className="w-6 h-6" />
                        Tomar Foto con Cámara
                      </div>
                    </button>

                    <div className="flex items-center gap-4">
                      <div className="flex-1 h-px bg-slate-700"></div>
                      <span className="text-slate-500 font-medium">o</span>
                      <div className="flex-1 h-px bg-slate-700"></div>
                    </div>

                    {/* Botón Upload *//*}
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

                    {/* Instrucciones *//*}
                    <div className="mt-8 bg-cyan-500/10 border border-cyan-500/30 rounded-2xl p-6">
                      <h4 className="font-semibold text-cyan-400 mb-4 flex items-center gap-2 text-lg">
                        <Info className="w-6 h-6" />
                        Instrucciones
                      </h4>
                      <ul className="text-sm text-slate-300 space-y-3">
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Captura o sube una imagen de estructura de concreto</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Puedes usar tu Raspberry Pi conectado para capturar fotos y ver streaming WebRTC en vivo</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>El sistema detecta grietas y analiza su patrón morfológico</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Recibe diagnóstico con causa probable y nivel de severidad</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Imagen Original *//*}
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

                    {/* Botones *//*}
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

                    {/* Processing *//*}
                    {isProcessing && (
                      <div className="bg-blue-500/10 border border-blue-500/30 rounded-2xl p-6">
                        <div className="flex items-center gap-4 mb-4">
                          <Loader className="w-10 h-10 text-blue-400 animate-spin" />
                          <div>
                            <p className="font-bold text-blue-400 text-xl">Procesando imagen...</p>
                            <p className="text-sm text-slate-400">
                              {useTTA ? 'UNet++ B8 + TTA (6x) + Análisis Morfológico' : 'UNet++ B8 Estándar'}
                            </p>
                          </div>
                        </div>
                        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                          <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 animate-pulse rounded-full w-full"></div>
                        </div>
                      </div>
                    )}

                    {/* Imagen Procesada *//*}
                    {processedImage && result && result.success && (
                      <div className="rounded-2xl overflow-hidden border border-slate-700">
                        <img
                          src={processedImage}
                          alt="Imagen procesada"
                          className="w-full h-80 object-contain bg-slate-900"
                          onError={(e) => {
                            console.error('Error cargando imagen procesada')
                            e.currentTarget.style.display = 'none'
                            setError('No se pudo cargar la imagen procesada')
                          }}
                        />
                        <div className="bg-slate-800 p-3 text-center border-t border-slate-700">
                          <p className="text-sm text-slate-400 font-medium">
                            Resultado Procesado
                            {result.procesamiento?.tta_usado && <span className="text-cyan-400"> • TTA {result.procesamiento.tta_transforms}x</span>}
                            {result.procesamiento?.cpu_optimized && <span className="text-green-400"> • CPU Optimizado</span>}
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Resultados (EL RESTO DEL CÓDIGO DE MÉTRICAS CONTINÚA IGUAL...) *//*}
                    {result && result.success && (
                      <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6">
                        {result.metricas.num_grietas_detectadas > 0 && result.metricas.porcentaje_grietas > 0 ? (
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

                            {/* ANÁLISIS MORFOLÓGICO *//*}
                            {result.metricas.analisis_morfologico && (
                              <div className="bg-gradient-to-br from-purple-500/10 to-pink-600/10 border border-purple-500/30 rounded-xl p-5 space-y-4">
                                <div className="flex items-center gap-3">
                                  <Compass className="w-6 h-6 text-purple-400" />
                                  <h5 className="font-bold text-purple-400 text-lg">Análisis Morfológico Avanzado</h5>
                                </div>

                                <div className="bg-slate-900/50 border border-purple-500/30 rounded-lg p-4">
                                  <div className="flex items-start gap-3 mb-2">
                                    <span className="text-2xl">{getPatronIcon(result.metricas.analisis_morfologico.patron_general)}</span>
                                    <div className="flex-1">
                                      <p className="font-semibold text-white capitalize text-lg">
                                        Patrón: {result.metricas.analisis_morfologico.patron_general.replace('_', ' ')}
                                      </p>
                                      <p className="text-sm text-slate-300 mt-1">
                                        {result.metricas.analisis_morfologico.descripcion_patron}
                                      </p>
                                    </div>
                                  </div>

                                  <div className="mt-3 bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
                                    <p className="text-sm text-orange-400">
                                      <strong>Causa probable:</strong> {result.metricas.analisis_morfologico.causa_probable}
                                    </p>
                                  </div>

                                  <div className={`mt-3 border rounded-lg p-3 ${getSeveridadBg(result.metricas.severidad)}`}>
                                    <p className={`text-sm font-medium ${getSeveridadColor(result.metricas.severidad)}`}>
                                      <strong>📋 Recomendación:</strong> {result.metricas.analisis_morfologico.recomendacion}
                                    </p>
                                  </div>
                                </div>

                                {/* Distribución Orientaciones *//*}
                                <div className="grid grid-cols-2 gap-2">
                                  {Object.entries(result.metricas.analisis_morfologico.distribucion_orientaciones || {}).map(([tipo, count]) => (
                                    count > 0 && (
                                      <div key={tipo} className={`border rounded-lg p-3 text-center ${getOrientacionColor(tipo)}`}>
                                        <p className="text-xs font-medium capitalize mb-1">{tipo}</p>
                                        <p className="text-2xl font-bold">{count}</p>
                                      </div>
                                    )
                                  ))}
                                </div>

                                {/* Top Grietas *//*}
                                {result.metricas.analisis_morfologico.grietas_principales &&
                                 result.metricas.analisis_morfologico.grietas_principales.length > 0 && (
                                  <div className="bg-slate-900/50 border border-slate-600 rounded-lg p-3">
                                    <p className="text-xs text-slate-400 mb-3 font-semibold flex items-center gap-2">
                                      🔍 Top {result.metricas.analisis_morfologico.grietas_principales.length} Grietas Analizadas
                                      <span className="text-cyan-400">
                                        (de {result.metricas.num_grietas_detectadas} totales)
                                      </span>
                                    </p>
                                    <div className="space-y-2">
                                      {result.metricas.analisis_morfologico.grietas_principales.slice(0, 5).map((grieta) => (
                                        <div key={grieta.id} className="flex items-center justify-between text-xs bg-slate-800 rounded p-2">
                                          <div className="flex-1">
                                            <span className="text-slate-300 font-semibold">
                                              #{grieta.id} • {grieta.orientacion}
                                            </span>
                                            {grieta.angulo_grados !== null && (
                                              <span className="text-slate-500 ml-2">
                                                ({safeToFixed(grieta.angulo_grados, 1)}°)
                                              </span>
                                            )}
                                          </div>
                                          <div className="text-right">
                                            <div className="text-cyan-400 font-bold">
                                              {safeToFixed(grieta.longitud_px, 0)}px
                                            </div>
                                            <div className="text-slate-500 text-xs">
                                              {safeToFixed(grieta.area_px, 0)}px²
                                            </div>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}

                            {/* Métricas Grid *//*}
                            <div className="grid grid-cols-2 gap-3">
                              {[
                                { label: 'Grietas detectadas', value: result.metricas.num_grietas_detectadas, icon: '🔍' },
                                { label: 'Cobertura', value: `${safeToFixed(result.metricas.porcentaje_grietas)}%`, icon: '📊' },
                                { label: 'Longitud máxima', value: `${safeToFixed(result.metricas.longitud_maxima_px, 0)} px`, icon: '📏' },
                                { label: 'Confianza', value: `${safeToFixed(result.metricas.confianza, 1)}%`, icon: '✓' },
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

                            {/* Info Procesamiento *//*}
                            {result.procesamiento && (
                              <div className="bg-slate-900/50 border border-slate-600 rounded-xl p-3 text-xs text-slate-400 space-y-1">
                                <p>
                                  🏗️ <strong>{result.procesamiento.architecture}</strong> + <strong>{result.procesamiento.encoder}</strong>
                                </p>
                                <p>
                                  ⚡ {result.procesamiento.tta_usado ? `TTA (${result.procesamiento.tta_transforms}x)` : 'Estándar'} •
                                  Umbral: {result.procesamiento.threshold} •
                                  Resolución: {result.procesamiento.target_size}px
                                </p>
                                {result.procesamiento.cpu_optimized && (
                                  <p className="text-green-400">
                                    🚀 CPU Optimizado ({result.procesamiento.cpu_threads} threads) •
                                    Max: {result.procesamiento.max_resolution}px •
                                    Salida: {result.procesamiento.output_format}
                                  </p>
                                )}
                                {result.procesamiento.original_dimensions && (
                                  <p>
                                    📐 Original: {result.procesamiento.original_dimensions.width}x{result.procesamiento.original_dimensions.height}px
                                  </p>
                                )}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="space-y-4">
                            <div className="flex items-center gap-3 mb-4">
                              <CheckCircle className="w-12 h-12 text-green-400" />
                              <div className="flex-1">
                                <h4 className="text-2xl font-bold text-white">{result.metricas.estado}</h4>
                                <p className="text-slate-400">Estructura en excelente estado</p>
                              </div>
                            </div>
                            <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4">
                              <p className="text-green-400 text-center font-medium flex items-center justify-center gap-2">
                                <CheckCircle className="w-5 h-5" />
                                Sin grietas significativas detectadas
                              </p>
                            </div>
                            {result.procesamiento && (
                              <div className="bg-slate-900/50 border border-slate-600 rounded-xl p-3 text-xs text-slate-400">
                                <p>
                                  {result.procesamiento.architecture} + {result.procesamiento.encoder} •
                                  {result.procesamiento.tta_usado ? ` TTA (${result.procesamiento.tta_transforms}x)` : ' Estándar'} •
                                  Confianza: {safeToFixed(result.metricas.confianza, 1)}%
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

            {/* COLUMNA DERECHA - INFO TÉCNICA (SIN CAMBIOS) *//*}
            <div className="space-y-6">
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-blue-500 to-indigo-600 p-2 rounded-xl">
                    <ImageIcon className="w-6 h-6 text-white" />
                  </div>
                  Tecnología IA v4.2
                </h3>

                <div className="space-y-4">
                  <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-5">
                    <h4 className="font-bold text-cyan-400 mb-3 text-lg">UNet++ EfficientNet-B8</h4>
                    <p className="text-slate-300 text-sm leading-relaxed mb-3">
                      Arquitectura encoder-decoder con Test-Time Augmentation y análisis morfológico condicional optimizado para CPU.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {['UNet++', 'EfficientNet-B8', 'TTA 6x', 'Morfología', 'CPU Opt', 'Raspberry Pi REST'].map((tag, idx) => (
                        <span key={idx} className="bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold px-3 py-1 rounded-full">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-indigo-500/10 to-blue-600/10 border border-indigo-500/30 rounded-xl p-5">
                    <h4 className="font-bold text-indigo-400 mb-3 text-lg">Patrones Detectados</h4>
                    <div className="space-y-2">
                      {[
                        { icon: '↔️', label: 'Horizontal', causa: 'Flexión, presión lateral', severidad: 'Media' },
                        { icon: '↕️', label: 'Vertical', causa: 'Cargas pesadas, asentamientos', severidad: 'Alta' },
                        { icon: '↗️', label: 'Diagonal', causa: 'Esfuerzos cortantes', severidad: 'Alta' },
                        { icon: '🗺️', label: 'Ramificada', causa: 'Contracción térmica', severidad: 'Baja' },
                      ].map((item, idx) => (
                        <div key={idx} className="bg-slate-900/50 rounded-lg p-3 flex items-start gap-3 hover:bg-slate-900 transition-all">
                          <span className="text-xl">{item.icon}</span>
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <p className="text-white font-semibold text-sm">{item.label}</p>
                              <span className={`text-xs px-2 py-0.5 rounded-full ${getSeveridadBg(item.severidad)}`}>
                                {item.severidad}
                              </span>
                            </div>
                            <p className="text-slate-400 text-xs">{item.causa}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                                   <div className="bg-gradient-to-br from-green-500/10 to-emerald-600/10 border border-green-500/30 rounded-xl p-5">
                    <h4 className="font-bold text-green-400 mb-3 text-lg flex items-center gap-2">
                      <Zap className="w-5 h-5" />
                      Optimizaciones v4.2
                    </h4>
                    <ul className="space-y-2 text-sm text-slate-300">
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Análisis morfológico solo si hay grietas detectadas</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Top 10 grietas más grandes (procesamiento rápido)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Resize inteligente (max 2048px sin pérdida)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Raspberry Pi con REST API (sin WebSocket)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Streaming WebRTC vía proxy HTTPS (MediaMTX)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Control remoto: iniciar/detener streaming</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Captura de fotos bajo demanda desde Raspberry Pi</span>
                      </li>
                    </ul>
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

*/












































/*





import { useState, useRef, useEffect } from 'react'
import { Camera, Upload, Image as ImageIcon, Zap, CheckCircle, AlertCircle, Loader, XCircle, AlertTriangle, Info, Settings, Compass, Wifi, Video, WifiOff, RefreshCw } from 'lucide-react'

// ═══════════════════════════════════════════════════════════════════════════
// INTERFACES (SIN CAMBIOS)
// ═══════════════════════════════════════════════════════════════════════════

interface AnalisisMorfologico {
  patron_general: string
  descripcion_patron: string
  causa_probable: string
  severidad_ajuste: number
  recomendacion: string
  distribucion_orientaciones: {
    horizontal: number
    vertical: number
    diagonal: number
    irregular: number
  }
  num_grietas_analizadas: number
  grietas_principales: Array<{
    id: number
    longitud_px: number
    area_px: number
    ancho_promedio_px: number
    angulo_grados: number | null
    orientacion: string
    aspect_ratio: number
    bbox: {
      x: number
      y: number
      width: number
      height: number
    }
  }>
}

interface Metricas {
  total_pixeles: number
  pixeles_con_grietas: number
  porcentaje_grietas: number
  num_grietas_detectadas: number
  longitud_total_px?: number
  longitud_promedio_px?: number
  longitud_maxima_px?: number
  ancho_promedio_px?: number
  severidad: string
  estado: string
  confianza: number
  confidence_max?: number
  confidence_mean?: number
  analisis_morfologico?: AnalisisMorfologico | null
}

interface Procesamiento {
  architecture: string
  encoder: string
  tta_usado: boolean
  tta_transforms: number
  threshold: number
  target_size: number
  cpu_optimized?: boolean
  cpu_threads?: number
  max_resolution?: number
  original_dimensions?: {
    width: number
    height: number
  }
  output_format?: string
}

interface PredictResponse {
  success: boolean
  metricas: Metricas
  imagen_overlay?: string
  timestamp: string
  procesamiento?: Procesamiento
  error?: string
}

// ✅ INTERFAZ ACTUALIZADA PARA REST API
interface RaspberryDevice {
  device_id: string
  type: string
  ip_local: string
  stream_port: number
  streaming_active: boolean
  stream_url_direct: string
  stream_url_proxy: string

  
  capabilities: string[]
  connected_at: string
  last_seen_ago: number
  has_photo?: boolean
  last_photo_time?: string
}

interface DevicesResponse {
  devices: RaspberryDevice[]
  total: number
  timestamp: string
}

const Pruebas = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [useTTA, setUseTTA] = useState(true)
  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)

  // ✅ ESTADOS PARA RASPBERRY PI (REST API)
  const [raspberryDevices, setRaspberryDevices] = useState<RaspberryDevice[]>([])
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null)
  const [isLoadingDevices, setIsLoadingDevices] = useState(false)
  const [isCapturingFromRaspi, setIsCapturingFromRaspi] = useState(false)
  const [showRaspberryPanel, setShowRaspberryPanel] = useState(false)
  const [streamUrl, setStreamUrl] = useState<string | null>(null)
  const [isStreamActive, setIsStreamActive] = useState(false)
  const [isStartingStream, setIsStartingStream] = useState(false)
  const [isStoppingStream, setIsStoppingStream] = useState(false)

  // ✅ ESTADOS PARA CONTRASEÑA
  const [showPasswordModal, setShowPasswordModal] = useState(false)
  const [passwordInput, setPasswordInput] = useState('')
  const [passwordError, setPasswordError] = useState<string | null>(null)
  const [authorizedDevice, setAuthorizedDevice] = useState<string | null>(null)
  const [actionType, setActionType] = useState<'capture' | 'stream' | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // ✅ API_URL CORRECTO (funciona con tu Nginx)
  const API_URL = import.meta.env.VITE_API_URL ||
                  (window.location.hostname === 'localhost'
                    ? 'http://localhost:5000'
                    : '')

  // ═══════════════════════════════════════════════════════════════════════════
  // ✅ CARGAR DISPOSITIVOS RASPBERRY PI (REST API)
  // ═══════════════════════════════════════════════════════════════════════════

  useEffect(() => {
    loadRaspberryDevices()
    const interval = setInterval(loadRaspberryDevices, 10000) // Cada 10s
    return () => clearInterval(interval)
  }, [])

  const loadRaspberryDevices = async () => {
    setIsLoadingDevices(true)
    try {
      console.log('📡 Cargando dispositivos desde:', `${API_URL}/api/rpi/devices`)
      
      const response = await fetch(`${API_URL}/api/rpi/devices`)
      
      if (response.ok) {
        const data: DevicesResponse = await response.json()
        setRaspberryDevices(data.devices)
        console.log('✅ Dispositivos conectados:', data.devices)
      } else {
        console.error('❌ Error al cargar dispositivos:', response.status)
      }
    } catch (err) {
      console.error('❌ Error al cargar dispositivos:', err)
    } finally {
      setIsLoadingDevices(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ✅ AUTENTICACIÓN PARA RASPBERRY PI
  // ═══════════════════════════════════════════════════════════════════════════

  const handleAuthForAction = (deviceId: string, type: 'capture' | 'stream') => {
    if (authorizedDevice === deviceId) {
      if (type === 'capture') {
        captureAndAnalyzeFromRaspberry(deviceId)
      } else if (type === 'stream') {
        if (isStreamActive && selectedDevice === deviceId) {
          stopStreaming(deviceId)
        } else {
          startStreaming(deviceId)
        }
      }
    } else {
      setSelectedDevice(deviceId)
      setActionType(type)
      setShowPasswordModal(true)
      setPasswordInput('')
      setPasswordError(null)
    }
  }

  const verifyPassword = () => {
    if (passwordInput === '2206' && selectedDevice) {
      setAuthorizedDevice(selectedDevice)
      setShowPasswordModal(false)
      if (actionType === 'capture') {
        captureAndAnalyzeFromRaspberry(selectedDevice)
      } else if (actionType === 'stream') {
        if (isStreamActive && selectedDevice) {
          stopStreaming(selectedDevice)
        } else {
          startStreaming(selectedDevice)
        }
      }
    } else {
      setPasswordError('Contraseña incorrecta')
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ✅ CAPTURAR FOTO DESDE RASPBERRY PI (REST API)
  // ═══════════════════════════════════════════════════════════════════════════

  const captureAndAnalyzeFromRaspberry = async (deviceId: string) => {
    setIsCapturingFromRaspi(true)
    setError(null)
    setSelectedDevice(deviceId)

    try {
      console.log(`📸 Solicitando captura a ${deviceId}...`)

      // ✅ ENVIAR COMANDO DE CAPTURA (REST API)
      const cmdResponse = await fetch(`${API_URL}/api/rpi/capture/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          resolution: '1920x1080',
          format: 'jpg'
        })
      })

      if (!cmdResponse.ok) {
        throw new Error('Error al enviar comando al Raspberry Pi')
      }

      console.log('✅ Comando enviado, esperando foto...')

      // ✅ ESPERAR A QUE EL RASPBERRY CAPTURE Y SUBA LA FOTO (3-5 segundos)
      await new Promise(resolve => setTimeout(resolve, 5000))

      // ✅ OBTENER ÚLTIMA FOTO DEL RASPBERRY PI
      const photoResponse = await fetch(`${API_URL}/api/rpi/latest-photo/${deviceId}`)

      if (!photoResponse.ok) {
        throw new Error('No se pudo obtener la foto del Raspberry Pi')
      }

      const photoData = await photoResponse.json()

      if (!photoData.success || !photoData.image) {
        throw new Error('Foto vacía o inválida')
      }

      console.log('✅ Foto recibida desde Raspberry Pi')

      // ✅ CONVERTIR BASE64 A FILE
      const base64Data = photoData.image.split(',')[1] || photoData.image
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'image/jpeg' })
      const file = new File([blob], `raspberry_${deviceId}_${Date.now()}.jpg`, { type: 'image/jpeg' })

      setSelectedFile(file)
      setSelectedImage(photoData.image.includes('data:') ? photoData.image : `data:image/jpeg;base64,${photoData.image}`)
      setResult(null)
      setProcessedImage(null)
      
      console.log('✅ Foto lista para análisis')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al capturar desde Raspberry Pi')
      console.error('❌ Error:', err)
    } finally {
      setIsCapturingFromRaspi(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ✅ INICIAR STREAMING DESDE RASPBERRY PI (REST API)
  // ═══════════════════════════════════════════════════════════════════════════

  const startStreaming = async (deviceId: string) => {
    setIsStartingStream(true)
    setError(null)
    setSelectedDevice(deviceId)

    try {
      console.log(`🎬 Iniciando streaming en ${deviceId}...`)

      // ✅ ENVIAR COMANDO DE START STREAMING
      const response = await fetch(`${API_URL}/api/rpi/streaming/start/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) {
        throw new Error('Error al iniciar streaming')
      }

      console.log('✅ Comando de inicio enviado')

      // ✅ ESPERAR A QUE EL RASPBERRY INICIE EL STREAMING
      await new Promise(resolve => setTimeout(resolve, 3000))

      // ✅ USAR PROXY DEL BACKEND PARA EVITAR MIXED CONTENT (HTTPS)
      const proxyUrl = `${API_URL}/api/stream/${deviceId}`
      
      setStreamUrl(proxyUrl)
      setIsStreamActive(true)

      console.log('✅ Streaming activo:', proxyUrl)

      // ✅ RECARGAR DISPOSITIVOS PARA ACTUALIZAR ESTADO
      loadRaspberryDevices()

    } catch (err) {
      console.error('❌ Error al iniciar streaming:', err)
      setError('No se pudo iniciar el streaming')
    } finally {
      setIsStartingStream(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ✅ DETENER STREAMING (REST API)
  // ═══════════════════════════════════════════════════════════════════════════

  const stopStreaming = async (deviceId: string) => {
    setIsStoppingStream(true)
    setError(null)

    try {
      console.log(`🛑 Deteniendo streaming en ${deviceId}...`)

      // ✅ ENVIAR COMANDO DE STOP STREAMING
      const response = await fetch(`${API_URL}/api/rpi/streaming/stop/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) {
        throw new Error('Error al detener streaming')
      }

      console.log('✅ Comando de detención enviado')

      setStreamUrl(null)
      setIsStreamActive(false)
      setSelectedDevice(null)

      // ✅ RECARGAR DISPOSITIVOS PARA ACTUALIZAR ESTADO
      await loadRaspberryDevices()

    } catch (err) {
      console.error('❌ Error al detener streaming:', err)
      setError('No se pudo detener el streaming')
    } finally {
      setIsStoppingStream(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // FUNCIONES DE CÁMARA LOCAL (SIN CAMBIOS)
  // ═══════════════════════════════════════════════════════════════════════════

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (file.size > 25 * 1024 * 1024) {
        setError('El archivo es demasiado grande. Máximo 25MB.')
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

  const openCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      })

      setStream(mediaStream)
      setIsCameraOpen(true)

      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream
        }
      }, 100)
    } catch (err) {
      console.error('Error al acceder a la cámara:', err)
      setError('No se pudo acceder a la cámara. Verifica los permisos.')
    }
  }

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    if (!context) return
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera_capture_${Date.now()}.jpg`, { type: 'image/jpeg' })
        setSelectedFile(file)
        setSelectedImage(URL.createObjectURL(blob))
        setResult(null)
        setProcessedImage(null)
        closeCamera()
      }
    }, 'image/jpeg', 0.95)
  }

  const closeCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    setIsCameraOpen(false)
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
      
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const contentType = response.headers.get('content-type')
        if (contentType?.includes('application/json')) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Error en la predicción')
        } else {
          throw new Error(`Error del servidor: ${response.status}`)
        }
      }
      
      const data: PredictResponse = await response.json()

      if (!data.success) {
        throw new Error(data.error || 'Error en la predicción')
      }
      
      setResult(data)
      if (data.imagen_overlay) {
        setProcessedImage(data.imagen_overlay)
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
    closeCamera()
    setStreamUrl(null)
    setIsStreamActive(false)
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // FUNCIONES DE UTILIDAD (SIN CAMBIOS)
  // ═══════════════════════════════════════════════════════════════════════════

  const getSeveridadColor = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta':
      case 'media-alta':
        return 'text-red-400'
      case 'media':
        return 'text-yellow-400'
      case 'baja':
        return 'text-green-400'
      case 'sin grietas':
        return 'text-slate-400'
      default:
        return 'text-slate-400'
    }
  }

  const getSeveridadBg = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta':
      case 'media-alta':
        return 'bg-red-500/10 border-red-500/30'
      case 'media':
        return 'bg-yellow-500/10 border-yellow-500/30'
      case 'baja':
        return 'bg-green-500/10 border-green-500/30'
      case 'sin grietas':
        return 'bg-slate-500/10 border-slate-500/30'
      default:
        return 'bg-slate-500/10 border-slate-500/30'
    }
  }

  const getSeveridadIcon = (severidad: string) => {
    switch (severidad.toLowerCase()) {
      case 'alta':
      case 'media-alta':
        return '🔴'
      case 'media':
        return '🟡'
      case 'baja':
        return '🟢'
      case 'sin grietas':
        return '✅'
      default:
        return '⚪'
    }
  }

  const getPatronIcon = (patron: string) => {
    switch (patron) {
      case 'horizontal': return '↔️'
      case 'vertical': return '↕️'
      case 'diagonal_escalera': return '↗️'
      case 'ramificada_mapa': return '🗺️'
      case 'mixto': return '🔀'
      case 'irregular': return '🌀'
      case 'superficial': return '📏'
      case 'sin_grietas': return '✅'
      default: return '❓'
    }
  }

  const getOrientacionColor = (orientacion: string) => {
    switch (orientacion) {
      case 'horizontal': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      case 'vertical': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'diagonal': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
      case 'irregular': return 'bg-purple-500/20 text-purple-400 border-purple-500/30'
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500/30'
    }
  }

  const safeToFixed = (value: number | undefined, decimals: number = 2): string => {
    return value !== undefined && value !== null ? value.toFixed(decimals) : '0.00'
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER (CON ADAPTACIONES PARA REST API Y CONTRASEÑA)
  // ═══════════════════════════════════════════════════════════════════════════

  return (
    <div className="pt-16 bg-slate-950 min-h-screen">
      <section className="relative py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full px-5 py-2 mb-6">
              <Camera className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">PRUEBAS EN VIVO v4.2 + RASPBERRY PI (REST API)</span>
            </div>
            
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">
              Prueba el Sistema
            </h2>
            
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              UNet++ EfficientNet-B8 + TTA + Análisis Morfológico + Raspberry Pi REST API
            </p>

            {/* ✅ CONTROLES SUPERIORES *//*}
            <div className="mt-8 flex flex-wrap justify-center gap-4">
              {/* TTA Toggle *//*}
              <div className="inline-flex items-center gap-4 bg-slate-800/50 border border-slate-700 rounded-full px-6 py-3">
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
                  {useTTA ? 'ACTIVADO (6x)' : 'DESACTIVADO'}
                </span>
              </div>

              {/* ✅ BOTÓN RASPBERRY PI *//*}
              <button
                onClick={() => {
                  setShowRaspberryPanel(!showRaspberryPanel)
                  if (!showRaspberryPanel) loadRaspberryDevices()
                }}
                className="inline-flex items-center gap-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white px-6 py-3 rounded-full font-semibold hover:scale-105 transition-all shadow-lg shadow-purple-500/50"
              >
                {raspberryDevices.length > 0 ? (
                  <>
                    <Wifi className="w-5 h-5" />
                    <span>{raspberryDevices.length} Raspberry Pi Conectados</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-5 h-5" />
                    <span>Buscar Raspberry Pi</span>
                  </>
                )}
              </button>
            </div>

            {/* ✅ PANEL RASPBERRY PI (REST API) *//*}
            {showRaspberryPanel && (
              <div className="mt-8 max-w-4xl mx-auto">
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-white flex items-center gap-3">
                      <Wifi className="w-6 h-6 text-purple-400" />
                      Dispositivos Raspberry Pi (REST API)
                    </h3>
                    <button
                      onClick={loadRaspberryDevices}
                      disabled={isLoadingDevices}
                      className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all disabled:opacity-50 flex items-center gap-2"
                    >
                      {isLoadingDevices ? (
                        <Loader className="w-4 h-4 animate-spin" />
                      ) : (
                        <RefreshCw className="w-4 h-4" />
                      )}
                      Actualizar
                    </button>
                  </div>

                  {raspberryDevices.length === 0 ? (
                    <div className="text-center py-8">
                      <WifiOff className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                      <p className="text-slate-400">No hay dispositivos conectados</p>
                      <p className="text-sm text-slate-500 mt-2">
                        Inicia el cliente bash en tu Raspberry Pi
                      </p>
                    </div>
                  ) : (
                    <div className="grid gap-4">
                      {raspberryDevices.map((device) => (
                        <div
                          key={device.device_id}
                          className={`bg-slate-900 border-2 rounded-xl p-5 transition-all ${
                            selectedDevice === device.device_id
                              ? 'border-purple-500 shadow-lg shadow-purple-500/50'
                              : 'border-slate-700 hover:border-slate-600'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-4">
                            <div>
                              <h4 className="text-lg font-bold text-white flex items-center gap-2">
                                <Camera className="w-5 h-5 text-purple-400" />
                                {device.device_id}
                              </h4>
                              <p className="text-sm text-slate-400 mt-1">
                                📍 {device.ip_local} • {device.type}
                              </p>
                              {device.has_photo && (
                                <p className="text-xs text-green-400 mt-1">
                                  📸 Última foto: {device.last_photo_time}
                                </p>
                              )}
                              <p className="text-xs text-slate-500 mt-1">
                                ⏱️ Última conexión: hace {device.last_seen_ago}s
                              </p>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                              <span className="text-xs text-green-400 font-semibold">ONLINE</span>
                            </div>
                          </div>

                          {/* Capabilities *//*}
                          <div className="flex flex-wrap gap-2 mb-4">
                            {device.capabilities.map((cap, idx) => (
                              <span
                                key={idx}
                                className="bg-purple-500/20 text-purple-400 text-xs font-semibold px-3 py-1 rounded-full border border-purple-500/30"
                              >
                                {cap}
                              </span>
                            ))}
                          </div>

                          {/* ✅ BOTONES DE ACCIÓN CON AUTENTICACIÓN *//*}
                          <div className="grid grid-cols-2 gap-3">
                            {/* Capturar Foto *//*}
                            <button
                              onClick={() => handleAuthForAction(device.device_id, 'capture')}
                              disabled={isCapturingFromRaspi}
                              className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                            >
                              {isCapturingFromRaspi && selectedDevice === device.device_id ? (
                                <Loader className="w-5 h-5 animate-spin" />
                              ) : (
                                <Camera className="w-5 h-5" />
                              )}
                              Capturar Foto
                            </button>

                            {/* Streaming Toggle *//*}
                            {device.capabilities.includes('streaming') && (
                              <button
                                onClick={() => handleAuthForAction(device.device_id, 'stream')}
                                disabled={isStartingStream || isStoppingStream}
                                className={`py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all flex items-center justify-center gap-2 disabled:opacity-50 ${
                                  device.streaming_active || (isStreamActive && selectedDevice === device.device_id)
                                    ? 'bg-red-500 hover:bg-red-600 text-white'
                                    : 'bg-gradient-to-r from-green-500 to-emerald-600 text-white'
                                }`}
                              >
                                {(isStartingStream || isStoppingStream) && selectedDevice === device.device_id ? (
                                  <Loader className="w-5 h-5 animate-spin" />
                                ) : device.streaming_active || (isStreamActive && selectedDevice === device.device_id) ? (
                                  <>
                                    <XCircle className="w-5 h-5" />
                                    Detener Stream
                                  </>
                                ) : (
                                  <>
                                    <Video className="w-5 h-5" />
                                    Iniciar Stream
                                  </>
                                )}
                              </button>
                            )}
                          </div>

                          {/* ✅ STREAMING IFRAME (PROXY BACKEND) *//*}
                          {isStreamActive && streamUrl && selectedDevice === device.device_id && (
                            <div className="mt-4 bg-black rounded-xl overflow-hidden border border-purple-500/50">
                              <iframe
                                src={streamUrl}
                                className="w-full aspect-video"
                                allow="camera; microphone"
                                title={`Stream ${device.device_id}`}
                                onError={() => {
                                  console.error('❌ Error al cargar streaming iframe')
                                  setError('No se pudo conectar al streaming')
                                }}
                              />
                              <div className="bg-slate-900 p-3 text-center">
                                <p className="text-sm text-purple-400 font-semibold flex items-center justify-center gap-2">
                                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                                  STREAMING EN VIVO • WebRTC • MediaMTX
                                </p>
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* ✅ MODAL DE CONTRASEÑA *//*}
            {showPasswordModal && (
              <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
                <div className="bg-slate-900 rounded-2xl p-8 max-w-md w-full border border-slate-700">
                  <h3 className="text-xl font-bold text-white mb-4">Acceso a Raspberry Pi</h3>
                  <p className="text-slate-400 mb-6">Ingresa la contraseña para acceder a la cámara/streaming</p>
                  
                  <input
                    type="password"
                    value={passwordInput}
                    onChange={(e) => setPasswordInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && verifyPassword()}
                    placeholder="Contraseña"
                    className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none mb-4"
                  />
                  
                  {passwordError && (
                    <p className="text-red-400 text-sm mb-4 flex items-center gap-2">
                      <AlertCircle className="w-4 h-4" />
                      {passwordError}
                    </p>
                  )}
                  
                  <div className="flex gap-4">
                    <button
                      onClick={verifyPassword}
                      className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 rounded-xl font-semibold hover:scale-105 transition-all"
                    >
                      Confirmar
                    </button>
                    <button
                      onClick={() => {
                        setShowPasswordModal(false)
                        setPasswordInput('')
                        setPasswordError(null)
                      }}
                      className="flex-1 bg-slate-700 text-slate-300 py-3 rounded-xl font-semibold hover:bg-slate-600 transition-all"
                    >
                      Cancelar
                    </button>
                  </div>
                </div>
              </div>
            )}

          </div>

          {/* ✅ ERROR DISPLAY *//*}
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

          {/* ✅ GRID PRINCIPAL (IMAGEN + RESULTADOS) *//*}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* COLUMNA IZQUIERDA - CAPTURA/ANÁLISIS *//*}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 rounded-3xl blur-2xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8 hover:border-cyan-500/50 transition-all duration-300">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-2 rounded-xl">
                    <Camera className="w-6 h-6 text-white" />
                  </div>
                  Captura de Imagen
                </h3>

                {/* ✅ MODAL CÁMARA *//*}
                {isCameraOpen && (
                  <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
                    <div className="relative max-w-4xl w-full">
                      <button
                        onClick={closeCamera}
                        className="absolute top-4 right-4 z-10 bg-red-500 hover:bg-red-600 text-white p-3 rounded-full transition-all"
                      >
                        <XCircle className="w-6 h-6" />
                      </button>

                      <div className="bg-slate-900 rounded-2xl overflow-hidden border border-slate-700">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          className="w-full h-auto"
                        />

                        <div className="p-6 flex justify-center gap-4">
                          <button
                            onClick={capturePhoto}
                            className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white px-8 py-4 rounded-xl font-semibold flex items-center gap-3 hover:scale-105 transition-all shadow-lg shadow-cyan-500/50"
                          >
                            <Camera className="w-6 h-6" />
                            Capturar Foto
                          </button>
                        </div>
                      </div>
                    </div>
                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                )}

                {!selectedImage ? (
                  <div className="space-y-4">
                    {/* Botón Cámara *//*}
                    <button
                      onClick={openCamera}
                      className="group/btn relative w-full overflow-hidden"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl blur-xl opacity-75 group-hover/btn:opacity-100 transition duration-300"></div>
                      <div className="relative bg-gradient-to-r from-purple-500 to-pink-600 text-white py-5 px-6 rounded-2xl font-semibold transition-all duration-300 flex items-center justify-center gap-3 shadow-lg shadow-purple-500/50 hover:scale-105">
                        <Camera className="w-6 h-6" />
                        Tomar Foto con Cámara
                      </div>
                    </button>

                    <div className="flex items-center gap-4">
                      <div className="flex-1 h-px bg-slate-700"></div>
                      <span className="text-slate-500 font-medium">o</span>
                      <div className="flex-1 h-px bg-slate-700"></div>
                    </div>

                    {/* Botón Upload *//*}
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

                    {/* Instrucciones *//*}
                    <div className="mt-8 bg-cyan-500/10 border border-cyan-500/30 rounded-2xl p-6">
                      <h4 className="font-semibold text-cyan-400 mb-4 flex items-center gap-2 text-lg">
                        <Info className="w-6 h-6" />
                        Instrucciones
                      </h4>
                      <ul className="text-sm text-slate-300 space-y-3">
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Captura o sube una imagen de estructura de concreto</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Puedes usar tu Raspberry Pi conectado para capturar fotos y ver streaming WebRTC en vivo</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>El sistema detecta grietas y analiza su patrón morfológico</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Recibe diagnóstico con causa probable y nivel de severidad</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Imagen Original *//*}
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

                    {/* Botones *//*}
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

                    {/* Processing *//*}
                    {isProcessing && (
                      <div className="bg-blue-500/10 border border-blue-500/30 rounded-2xl p-6">
                        <div className="flex items-center gap-4 mb-4">
                          <Loader className="w-10 h-10 text-blue-400 animate-spin" />
                          <div>
                            <p className="font-bold text-blue-400 text-xl">Procesando imagen...</p>
                            <p className="text-sm text-slate-400">
                              {useTTA ? 'UNet++ B8 + TTA (6x) + Análisis Morfológico' : 'UNet++ B8 Estándar'}
                            </p>
                          </div>
                        </div>
                        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                          <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 animate-pulse rounded-full w-full"></div>
                        </div>
                      </div>
                    )}

                    {/* Imagen Procesada *//*}
                    {processedImage && result && result.success && (
                      <div className="rounded-2xl overflow-hidden border border-slate-700">
                        <img
                          src={processedImage}
                          alt="Imagen procesada"
                          className="w-full h-80 object-contain bg-slate-900"
                          onError={(e) => {
                            console.error('Error cargando imagen procesada')
                            e.currentTarget.style.display = 'none'
                            setError('No se pudo cargar la imagen procesada')
                          }}
                        />
                        <div className="bg-slate-800 p-3 text-center border-t border-slate-700">
                          <p className="text-sm text-slate-400 font-medium">
                            Resultado Procesado
                            {result.procesamiento?.tta_usado && <span className="text-cyan-400"> • TTA {result.procesamiento.tta_transforms}x</span>}
                            {result.procesamiento?.cpu_optimized && <span className="text-green-400"> • CPU Optimizado</span>}
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Resultados (EL RESTO DEL CÓDIGO DE MÉTRICAS CONTINÚA IGUAL...) *//*}
                    {result && result.success && (
                      <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6">
                        {result.metricas.num_grietas_detectadas > 0 && result.metricas.porcentaje_grietas > 0 ? (
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

                            {/* ANÁLISIS MORFOLÓGICO *//*}
                            {result.metricas.analisis_morfologico && (
                              <div className="bg-gradient-to-br from-purple-500/10 to-pink-600/10 border border-purple-500/30 rounded-xl p-5 space-y-4">
                                <div className="flex items-center gap-3">
                                  <Compass className="w-6 h-6 text-purple-400" />
                                  <h5 className="font-bold text-purple-400 text-lg">Análisis Morfológico Avanzado</h5>
                                </div>

                                <div className="bg-slate-900/50 border border-purple-500/30 rounded-lg p-4">
                                  <div className="flex items-start gap-3 mb-2">
                                    <span className="text-2xl">{getPatronIcon(result.metricas.analisis_morfologico.patron_general)}</span>
                                    <div className="flex-1">
                                      <p className="font-semibold text-white capitalize text-lg">
                                        Patrón: {result.metricas.analisis_morfologico.patron_general.replace('_', ' ')}
                                      </p>
                                      <p className="text-sm text-slate-300 mt-1">
                                        {result.metricas.analisis_morfologico.descripcion_patron}
                                      </p>
                                    </div>
                                  </div>

                                  <div className="mt-3 bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
                                    <p className="text-sm text-orange-400">
                                      <strong>Causa probable:</strong> {result.metricas.analisis_morfologico.causa_probable}
                                    </p>
                                  </div>

                                  <div className={`mt-3 border rounded-lg p-3 ${getSeveridadBg(result.metricas.severidad)}`}>
                                    <p className={`text-sm font-medium ${getSeveridadColor(result.metricas.severidad)}`}>
                                      <strong>📋 Recomendación:</strong> {result.metricas.analisis_morfologico.recomendacion}
                                    </p>
                                  </div>
                                </div>

                                {/* Distribución Orientaciones *//*}
                                <div className="grid grid-cols-2 gap-2">
                                  {Object.entries(result.metricas.analisis_morfologico.distribucion_orientaciones || {}).map(([tipo, count]) => (
                                    count > 0 && (
                                      <div key={tipo} className={`border rounded-lg p-3 text-center ${getOrientacionColor(tipo)}`}>
                                        <p className="text-xs font-medium capitalize mb-1">{tipo}</p>
                                        <p className="text-2xl font-bold">{count}</p>
                                      </div>
                                    )
                                  ))}
                                </div>

                                {/* Top Grietas *//*}
                                {result.metricas.analisis_morfologico.grietas_principales &&
                                 result.metricas.analisis_morfologico.grietas_principales.length > 0 && (
                                  <div className="bg-slate-900/50 border border-slate-600 rounded-lg p-3">
                                    <p className="text-xs text-slate-400 mb-3 font-semibold flex items-center gap-2">
                                      🔍 Top {result.metricas.analisis_morfologico.grietas_principales.length} Grietas Analizadas
                                      <span className="text-cyan-400">
                                        (de {result.metricas.num_grietas_detectadas} totales)
                                      </span>
                                    </p>
                                    <div className="space-y-2">
                                      {result.metricas.analisis_morfologico.grietas_principales.slice(0, 5).map((grieta) => (
                                        <div key={grieta.id} className="flex items-center justify-between text-xs bg-slate-800 rounded p-2">
                                          <div className="flex-1">
                                            <span className="text-slate-300 font-semibold">
                                              #{grieta.id} • {grieta.orientacion}
                                            </span>
                                            {grieta.angulo_grados !== null && (
                                              <span className="text-slate-500 ml-2">
                                                ({safeToFixed(grieta.angulo_grados, 1)}°)
                                              </span>
                                            )}
                                          </div>
                                          <div className="text-right">
                                            <div className="text-cyan-400 font-bold">
                                              {safeToFixed(grieta.longitud_px, 0)}px
                                            </div>
                                            <div className="text-slate-500 text-xs">
                                              {safeToFixed(grieta.area_px, 0)}px²
                                            </div>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}

                            {/* Métricas Grid *//*}
                            <div className="grid grid-cols-2 gap-3">
                              {[
                                { label: 'Grietas detectadas', value: result.metricas.num_grietas_detectadas, icon: '🔍' },
                                { label: 'Cobertura', value: `${safeToFixed(result.metricas.porcentaje_grietas)}%`, icon: '📊' },
                                { label: 'Longitud máxima', value: `${safeToFixed(result.metricas.longitud_maxima_px, 0)} px`, icon: '📏' },
                                { label: 'Confianza', value: `${safeToFixed(result.metricas.confianza, 1)}%`, icon: '✓' },
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

                            {/* Info Procesamiento *//*}
                            {result.procesamiento && (
                              <div className="bg-slate-900/50 border border-slate-600 rounded-xl p-3 text-xs text-slate-400 space-y-1">
                                <p>
                                  🏗️ <strong>{result.procesamiento.architecture}</strong> + <strong>{result.procesamiento.encoder}</strong>
                                </p>
                                <p>
                                  ⚡ {result.procesamiento.tta_usado ? `TTA (${result.procesamiento.tta_transforms}x)` : 'Estándar'} •
                                  Umbral: {result.procesamiento.threshold} •
                                  Resolución: {result.procesamiento.target_size}px
                                </p>
                                {result.procesamiento.cpu_optimized && (
                                  <p className="text-green-400">
                                    🚀 CPU Optimizado ({result.procesamiento.cpu_threads} threads) •
                                    Max: {result.procesamiento.max_resolution}px •
                                    Salida: {result.procesamiento.output_format}
                                  </p>
                                )}
                                {result.procesamiento.original_dimensions && (
                                  <p>
                                    📐 Original: {result.procesamiento.original_dimensions.width}x{result.procesamiento.original_dimensions.height}px
                                  </p>
                                )}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="space-y-4">
                            <div className="flex items-center gap-3 mb-4">
                              <CheckCircle className="w-12 h-12 text-green-400" />
                              <div className="flex-1">
                                <h4 className="text-2xl font-bold text-white">{result.metricas.estado}</h4>
                                <p className="text-slate-400">Estructura en excelente estado</p>
                              </div>
                            </div>
                            <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4">
                              <p className="text-green-400 text-center font-medium flex items-center justify-center gap-2">
                                <CheckCircle className="w-5 h-5" />
                                Sin grietas significativas detectadas
                              </p>
                            </div>
                            {result.procesamiento && (
                              <div className="bg-slate-900/50 border border-slate-600 rounded-xl p-3 text-xs text-slate-400">
                                <p>
                                  {result.procesamiento.architecture} + {result.procesamiento.encoder} •
                                  {result.procesamiento.tta_usado ? ` TTA (${result.procesamiento.tta_transforms}x)` : ' Estándar'} •
                                  Confianza: {safeToFixed(result.metricas.confianza, 1)}%
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

            {/* COLUMNA DERECHA - INFO TÉCNICA (SIN CAMBIOS) *//*}
            <div className="space-y-6">
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-blue-500 to-indigo-600 p-2 rounded-xl">
                    <ImageIcon className="w-6 h-6 text-white" />
                  </div>
                  Tecnología IA v4.2
                </h3>

                <div className="space-y-4">
                  <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-5">
                    <h4 className="font-bold text-cyan-400 mb-3 text-lg">UNet++ EfficientNet-B8</h4>
                    <p className="text-slate-300 text-sm leading-relaxed mb-3">
                      Arquitectura encoder-decoder con Test-Time Augmentation y análisis morfológico condicional optimizado para CPU.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {['UNet++', 'EfficientNet-B8', 'TTA 6x', 'Morfología', 'CPU Opt', 'Raspberry Pi REST'].map((tag, idx) => (
                        <span key={idx} className="bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold px-3 py-1 rounded-full">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-indigo-500/10 to-blue-600/10 border border-indigo-500/30 rounded-xl p-5">
                    <h4 className="font-bold text-indigo-400 mb-3 text-lg">Patrones Detectados</h4>
                    <div className="space-y-2">
                      {[
                        { icon: '↔️', label: 'Horizontal', causa: 'Flexión, presión lateral', severidad: 'Media' },
                        { icon: '↕️', label: 'Vertical', causa: 'Cargas pesadas, asentamientos', severidad: 'Alta' },
                        { icon: '↗️', label: 'Diagonal', causa: 'Esfuerzos cortantes', severidad: 'Alta' },
                        { icon: '🗺️', label: 'Ramificada', causa: 'Contracción térmica', severidad: 'Baja' },
                      ].map((item, idx) => (
                        <div key={idx} className="bg-slate-900/50 rounded-lg p-3 flex items-start gap-3 hover:bg-slate-900 transition-all">
                          <span className="text-xl">{item.icon}</span>
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <p className="text-white font-semibold text-sm">{item.label}</p>
                              <span className={`text-xs px-2 py-0.5 rounded-full ${getSeveridadBg(item.severidad)}`}>
                                {item.severidad}
                              </span>
                            </div>
                            <p className="text-slate-400 text-xs">{item.causa}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-green-500/10 to-emerald-600/10 border border-green-500/30 rounded-xl p-5">
                    <h4 className="font-bold text-green-400 mb-3 text-lg flex items-center gap-2">
                      <Zap className="w-5 h-5" />
                      Optimizaciones v4.2
                    </h4>
                    <ul className="space-y-2 text-sm text-slate-300">
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Análisis morfológico solo si hay grietas detectadas</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Top 10 grietas más grandes (procesamiento rápido)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Resize inteligente (max 2048px sin pérdida)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Raspberry Pi con REST API (sin WebSocket)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Streaming WebRTC vía proxy HTTPS (MediaMTX)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Control remoto: iniciar/detener streaming</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">✓</span>
                        <span>Captura de fotos bajo demanda desde Raspberry Pi</span>
                      </li>
                    </ul>
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


*/'use client'

import { useState, useRef, useEffect } from 'react'
import { 
  Camera, Upload, Image as ImageIcon, Zap, 
  AlertCircle, Loader, XCircle, 
  AlertTriangle, Settings, 
  Wifi, Video, WifiOff, RefreshCw, Globe 
} from 'lucide-react'

// ═══════════════════════════════════════════════════════════════════════════
// INTERFACES
// ═══════════════════════════════════════════════════════════════════════════
interface AnalisisMorfologico {
  patron_general: string
  descripcion_patron: string
  causa_probable: string
  severidad_ajuste: number
  recomendacion: string
  distribucion_orientaciones: {
    horizontal: number
    vertical: number
    diagonal: number
    irregular: number
  }
  num_grietas_analizadas: number
  grietas_principales: Array<{
    id: number
    longitud_px: number
    area_px: number
    ancho_promedio_px: number
    angulo_grados: number | null
    orientacion: string
    aspect_ratio: number
    bbox: {
      x: number
      y: number
      width: number
      height: number
    }
  }>
}

interface Metricas {
  total_pixeles: number
  pixeles_con_grietas: number
  porcentaje_grietas: number
  num_grietas_detectadas: number
  longitud_total_px?: number
  longitud_promedio_px?: number
  longitud_maxima_px?: number
  ancho_promedio_px?: number
  severidad: string
  estado: string
  confianza: number
  confidence_max?: number
  confidence_mean?: number
  analisis_morfologico?: AnalisisMorfologico | null
}

interface Procesamiento {
  architecture: string
  encoder: string
  tta_usado: boolean
  tta_transforms: number
  threshold: number
  target_size: number
  cpu_optimized?: boolean
  cpu_threads?: number
  max_resolution?: number
  original_dimensions?: {
    width: number
    height: number
  }
  output_format?: string
}

interface PredictResponse {
  success: boolean
  metricas: Metricas
  imagen_overlay?: string
  timestamp: string
  procesamiento?: Procesamiento
  error?: string
}

interface RaspberryDevice {
  device_id: string
  type: string
  ip_local: string
  stream_port: number
  streaming_active: boolean
  stream_url_direct: string
  stream_url_proxy: string
  stream_url_public?: string
  tunnel_type?: string
  capabilities: string[]
  connected_at: string
  last_seen_ago: number
  has_photo?: boolean
  last_photo_time?: string
}

interface DevicesResponse {
  devices: RaspberryDevice[]
  total: number
  timestamp: string
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPONENTE PRINCIPAL
// ═══════════════════════════════════════════════════════════════════════════
const Pruebas = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [useTTA, setUseTTA] = useState(true)
  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [raspberryDevices, setRaspberryDevices] = useState<RaspberryDevice[]>([])
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null)
  const [isLoadingDevices, setIsLoadingDevices] = useState(false)
  const [isCapturingFromRaspi, setIsCapturingFromRaspi] = useState(false)
  const [showRaspberryPanel, setShowRaspberryPanel] = useState(false)
  const [streamUrl, setStreamUrl] = useState<string | null>(null)
  const [isStreamActive, setIsStreamActive] = useState(false)
  const [isStartingStream, setIsStartingStream] = useState(false)
  const [isStoppingStream, setIsStoppingStream] = useState(false)
  const [showPasswordModal, setShowPasswordModal] = useState(false)
  const [passwordInput, setPasswordInput] = useState('')
  const [passwordError, setPasswordError] = useState<string | null>(null)
  const [authorizedDevice, setAuthorizedDevice] = useState<string | null>(null)
  const [actionType, setActionType] = useState<'capture' | 'stream' | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const API_URL = import.meta.env.VITE_API_URL ||
                  (window.location.hostname === 'localhost'
                    ? 'http://localhost:5000'
                    : '')

  // ═══════════════════════════════════════════════════════════════════════════
  // CARGAR DISPOSITIVOS
  // ═══════════════════════════════════════════════════════════════════════════
  useEffect(() => {
    loadRaspberryDevices()
    const interval = setInterval(loadRaspberryDevices, 10000)
    return () => clearInterval(interval)
  }, [])

  const loadRaspberryDevices = async () => {
    setIsLoadingDevices(true)
    try {
      const response = await fetch(`${API_URL}/api/rpi/devices`)
      if (response.ok) {
        const data: DevicesResponse = await response.json()
        setRaspberryDevices(data.devices)
      }
    } catch (err) {
      console.error('Error al cargar dispositivos:', err)
    } finally {
      setIsLoadingDevices(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // AUTENTICACIÓN
  // ═══════════════════════════════════════════════════════════════════════════
  const handleAuthForAction = (deviceId: string, type: 'capture' | 'stream') => {
    if (authorizedDevice === deviceId) {
      type === 'capture' ? captureAndAnalyzeFromRaspberry(deviceId) : 
      (isStreamActive && selectedDevice === deviceId ? stopStreaming(deviceId) : startStreaming(deviceId))
    } else {
      setSelectedDevice(deviceId)
      setActionType(type)
      setShowPasswordModal(true)
      setPasswordInput('')
      setPasswordError(null)
    }
  }

  const verifyPassword = () => {
    if (passwordInput === '2206' && selectedDevice) {
      setAuthorizedDevice(selectedDevice)
      setShowPasswordModal(false)
      actionType === 'capture' ? captureAndAnalyzeFromRaspberry(selectedDevice) :
      (isStreamActive && selectedDevice ? stopStreaming(selectedDevice) : startStreaming(selectedDevice))
    } else {
      setPasswordError('Contraseña incorrecta')
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // CAPTURAR FOTO DESDE RASPBERRY
  // ═══════════════════════════════════════════════════════════════════════════
  const captureAndAnalyzeFromRaspberry = async (deviceId: string) => {
    setIsCapturingFromRaspi(true)
    setError(null)
    setSelectedDevice(deviceId)
    try {
      const cmdResponse = await fetch(`${API_URL}/api/rpi/capture/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution: '1920x1080', format: 'jpg' })
      })
      if (!cmdResponse.ok) throw new Error('Error al enviar comando')

      await new Promise(resolve => setTimeout(resolve, 5000))
      const photoResponse = await fetch(`${API_URL}/api/rpi/latest-photo/${deviceId}`)
      if (!photoResponse.ok) throw new Error('No se pudo obtener la foto')

      const photoData = await photoResponse.json()
      if (!photoData.success || !photoData.image) throw new Error('Foto vacía')

      const base64Data = photoData.image.split(',')[1] || photoData.image
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'image/jpeg' })
      const file = new File([blob], `raspberry_${deviceId}_${Date.now()}.jpg`, { type: 'image/jpeg' })

      setSelectedFile(file)
      setSelectedImage(photoData.image.includes('data:') ? photoData.image : `data:image/jpeg;base64,${photoData.image}`)
      setResult(null)
      setProcessedImage(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al capturar')
    } finally {
      setIsCapturingFromRaspi(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // STREAMING (Cloudflare Tunnel)
  // ═══════════════════════════════════════════════════════════════════════════
  const startStreaming = async (deviceId: string) => {
    setIsStartingStream(true)
    setError(null)
    setSelectedDevice(deviceId)
    try {
      const response = await fetch(`${API_URL}/api/rpi/streaming/start/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      if (!response.ok) throw new Error('Error al iniciar streaming')

      await new Promise(resolve => setTimeout(resolve, 8000))
      await loadRaspberryDevices()

      const updatedDevice = raspberryDevices.find(d => d.device_id === deviceId)
      if (!updatedDevice) throw new Error('Dispositivo no encontrado')

      const finalStreamUrl = updatedDevice.stream_url_public || `${API_URL}/api/stream/${deviceId}`
      setStreamUrl(finalStreamUrl)
      setIsStreamActive(true)
    } catch (err) {
      setError('No se pudo iniciar el streaming')
    } finally {
      setIsStartingStream(false)
    }
  }

  const stopStreaming = async (deviceId: string) => {
    setIsStoppingStream(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/api/rpi/streaming/stop/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      if (!response.ok) throw new Error('Error al detener streaming')

      setStreamUrl(null)
      setIsStreamActive(false)
      setSelectedDevice(null)
      await loadRaspberryDevices()
    } catch (err) {
      setError('No se pudo detener el streaming')
    } finally {
      setIsStoppingStream(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // CÁMARA LOCAL
  // ═══════════════════════════════════════════════════════════════════════════
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (file.size > 25 * 1024 * 1024) {
      setError('El archivo es demasiado grande. Máximo 25MB.')
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

  const openCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }
      })
      setStream(mediaStream)
      setIsCameraOpen(true)
      setTimeout(() => {
        if (videoRef.current) videoRef.current.srcObject = mediaStream
      }, 100)
    } catch (err) {
      setError('No se pudo acceder a la cámara. Verifica los permisos.')
    }
  }

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    if (!context) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera_capture_${Date.now()}.jpg`, { type: 'image/jpeg' })
        setSelectedFile(file)
        setSelectedImage(URL.createObjectURL(blob))
        setResult(null)
        setProcessedImage(null)
        closeCamera()
      }
    }, 'image/jpeg', 0.95)
  }

  const closeCamera = () => {
    stream?.getTracks().forEach(track => track.stop())
    setStream(null)
    setIsCameraOpen(false)
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

      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const contentType = response.headers.get('content-type')
        if (contentType?.includes('application/json')) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Error en la predicción')
        } else {
          throw new Error(`Error del servidor: ${response.status}`)
        }
      }

      const data: PredictResponse = await response.json()
      if (!data.success) throw new Error(data.error || 'Error en la predicción')

      setResult(data)
      if (data.imagen_overlay) setProcessedImage(data.imagen_overlay)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido')
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
    closeCamera()
    setStreamUrl(null)
    setIsStreamActive(false)
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════
  return (
    <div className="pt-16 bg-slate-950 min-h-screen">
      <section className="relative py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full px-5 py-2 mb-6">
              <Camera className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">PRUEBAS EN VIVO v5.0 + CLOUDFLARE TUNNEL</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">
              Prueba el Sistema
            </h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              UNet++ EfficientNet-B8 + TTA + Análisis Morfológico + Raspberry Pi + Cloudflare Tunnel
            </p>

            {/* CONTROLES SUPERIORES */}
            <div className="mt-8 flex flex-wrap justify-center gap-4">
              <div className="inline-flex items-center gap-4 bg-slate-800/50 border border-slate-700 rounded-full px-6 py-3">
                <Settings className="w-5 h-5 text-slate-400" />
                <span className="text-slate-300 font-medium">Test-Time Augmentation</span>
                <button
                  onClick={() => setUseTTA(!useTTA)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${useTTA ? 'bg-cyan-500' : 'bg-slate-600'}`}
                >
                  <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${useTTA ? 'translate-x-6' : 'translate-x-1'}`} />
                </button>
                <span className={`text-sm font-semibold ${useTTA ? 'text-cyan-400' : 'text-slate-500'}`}>
                  {useTTA ? 'ACTIVADO (6x)' : 'DESACTIVADO'}
                </span>
              </div>
              <button
                onClick={() => {
                  setShowRaspberryPanel(!showRaspberryPanel)
                  if (!showRaspberryPanel) loadRaspberryDevices()
                }}
                className="inline-flex items-center gap-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white px-6 py-3 rounded-full font-semibold hover:scale-105 transition-all shadow-lg shadow-purple-500/50"
              >
                {raspberryDevices.length > 0 ? (
                  <>
                    <Wifi className="w-5 h-5" />
                    <span>{raspberryDevices.length} Raspberry Pi Conectados</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-5 h-5" />
                    <span>Buscar Raspberry Pi</span>
                  </>
                )}
              </button>
            </div>

            {/* PANEL RASPBERRY PI */}
            {showRaspberryPanel && (
              <div className="mt-8 max-w-4xl mx-auto">
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-white flex items-center gap-3">
                      <Wifi className="w-6 h-6 text-purple-400" />
                      Dispositivos Raspberry Pi + Cloudflare Tunnel
                    </h3>
                    <button
                      onClick={loadRaspberryDevices}
                      disabled={isLoadingDevices}
                      className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all disabled:opacity-50 flex items-center gap-2"
                    >
                      {isLoadingDevices ? <Loader className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                      Actualizar
                    </button>
                  </div>

                  {raspberryDevices.length === 0 ? (
                    <div className="text-center py-8">
                      <WifiOff className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                      <p className="text-slate-400">No hay dispositivos conectados</p>
                    </div>
                  ) : (
                    <div className="grid gap-4">
                      {raspberryDevices.map((device) => (
                        <div
                          key={device.device_id}
                          className={`bg-slate-900 border-2 rounded-xl p-5 transition-all ${
                            selectedDevice === device.device_id ? 'border-purple-500 shadow-lg shadow-purple-500/50' : 'border-slate-700 hover:border-slate-600'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-4">
                            <div>
                              <h4 className="text-lg font-bold text-white flex items-center gap-2">
                                <Camera className="w-5 h-5 text-purple-400" />
                                {device.device_id}
                              </h4>
                              <p className="text-sm text-slate-400 mt-1">
                                {device.ip_local} • {device.type}
                              </p>
                              {device.stream_url_public && (
                                <p className="text-xs text-cyan-400 mt-1 flex items-center gap-1">
                                  <Globe className="w-3 h-3" />
                                  Cloudflare Tunnel activo
                                  {device.tunnel_type && <span className="text-slate-500">• {device.tunnel_type}</span>}
                                </p>
                              )}
                              {device.has_photo && (
                                <p className="text-xs text-green-400 mt-1">
                                  Última foto: {device.last_photo_time}
                                </p>
                              )}
                              <p className="text-xs text-slate-500 mt-1">
                                Última conexión: hace {device.last_seen_ago}s
                              </p>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                              <span className="text-xs text-green-400 font-semibold">ONLINE</span>
                            </div>
                          </div>

                          <div className="flex flex-wrap gap-2 mb-4">
                            {device.capabilities.map((cap, idx) => (
                              <span key={idx} className="bg-purple-500/20 text-purple-400 text-xs font-semibold px-3 py-1 rounded-full border border-purple-500/30">
                                {cap}
                              </span>
                            ))}
                          </div>

                          <div className="grid grid-cols-2 gap-3">
                            <button
                              onClick={() => handleAuthForAction(device.device_id, 'capture')}
                              disabled={isCapturingFromRaspi}
                              className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                            >
                              {isCapturingFromRaspi && selectedDevice === device.device_id ? (
                                <Loader className="w-5 h-5 animate-spin" />
                              ) : (
                                <Camera className="w-5 h-5" />
                              )}
                              Capturar Foto
                            </button>

                            {device.capabilities.includes('streaming') && (
                              <button
                                onClick={() => handleAuthForAction(device.device_id, 'stream')}
                                disabled={isStartingStream || isStoppingStream}
                                className={`py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all flex items-center justify-center gap-2 disabled:opacity-50 ${
                                  device.streaming_active || (isStreamActive && selectedDevice === device.device_id)
                                    ? 'bg-red-500 hover:bg-red-600 text-white'
                                    : 'bg-gradient-to-r from-green-500 to-emerald-600 text-white'
                                }`}
                              >
                                {(isStartingStream || isStoppingStream) && selectedDevice === device.device_id ? (
                                  <Loader className="w-5 h-5 animate-spin" />
                                ) : device.streaming_active || (isStreamActive && selectedDevice === device.device_id) ? (
                                  <>
                                    <XCircle className="w-5 h-5" />
                                    Detener Stream
                                  </>
                                ) : (
                                  <>
                                    <Video className="w-5 h-5" />
                                    Iniciar Stream
                                  </>
                                )}
                              </button>
                            )}
                          </div>

                          {isStreamActive && streamUrl && selectedDevice === device.device_id && (
                            <div className="mt-4 bg-black rounded-xl overflow-hidden border border-purple-500/50">
                              <iframe
                                src={streamUrl}
                                className="w-full aspect-video"
                                allow="camera; microphone"
                                title={`Stream ${device.device_id}`}
                                onLoad={() => console.log('Iframe cargado:', streamUrl)}
                                onError={() => setError('Error al cargar stream')}
                              />
                              <div className="bg-slate-900 p-3 text-center">
                                <p className="text-sm text-purple-400 font-semibold flex items-center justify-center gap-2">
                                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                                  STREAMING EN VIVO
                                  {device.stream_url_public && (
                                    <span className="text-cyan-400 ml-2 flex items-center gap-1">
                                      <Globe className="w-3 h-3" />
                                      Cloudflare
                                    </span>
                                  )}
                                </p>
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* MODAL CONTRASEÑA */}
            {showPasswordModal && (
              <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
                <div className="bg-slate-900 rounded-2xl p-8 max-w-md w-full border border-slate-700">
                  <h3 className="text-xl font-bold text-white mb-4">Acceso a Raspberry Pi</h3>
                  <p className="text-slate-400 mb-6">Ingresa la contraseña</p>
                  <input
                    type="password"
                    value={passwordInput}
                    onChange={(e) => setPasswordInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && verifyPassword()}
                    placeholder="Contraseña"
                    className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none mb-4"
                  />
                  {passwordError && (
                    <p className="text-red-400 text-sm mb-4 flex items-center gap-2">
                      <AlertCircle className="w-4 h-4" />
                      {passwordError}
                    </p>
                  )}
                  <div className="flex gap-4">
                    <button onClick={verifyPassword} className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 rounded-xl font-semibold hover:scale-105 transition-all">
                      Confirmar
                    </button>
                    <button onClick={() => { setShowPasswordModal(false); setPasswordInput(''); setPasswordError(null); }} className="flex-1 bg-slate-700 text-slate-300 py-3 rounded-xl font-semibold hover:bg-slate-600 transition-all">
                      Cancelar
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ERROR */}
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
                  <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300">
                    <XCircle className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* GRID PRINCIPAL */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* IZQUIERDA */}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 rounded-3xl blur-2xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8 hover:border-cyan-500/50 transition-all duration-300">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-2 rounded-xl">
                    <Camera className="w-6 h-6 text-white" />
                  </div>
                  Captura de Imagen
                </h3>

                {isCameraOpen && (
                  <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
                    <div className="relative max-w-4xl w-full">
                      <button onClick={closeCamera} className="absolute top-4 right-4 z-10 bg-red-500 hover:bg-red-600 text-white p-3 rounded-full">
                        <XCircle className="w-6 h-6" />
                      </button>
                      <div className="bg-slate-900 rounded-2xl overflow-hidden border border-slate-700">
                        <video ref={videoRef} autoPlay playsInline className="w-full h-auto" />
                        <div className="p-6 flex justify-center gap-4">
                          <button onClick={capturePhoto} className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white px-8 py-4 rounded-xl font-semibold flex items-center gap-3 hover:scale-105 transition-all shadow-lg shadow-cyan-500/50">
                            <Camera className="w-6 h-6" />
                            Capturar Foto
                          </button>
                        </div>
                      </div>
                    </div>
                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                )}

                {!selectedImage ? (
                  <div className="space-y-4">
                    <button onClick={openCamera} className="group/btn relative w-full overflow-hidden">
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl blur-xl opacity-75 group-hover/btn:opacity-100 transition duration-300"></div>
                      <div className="relative bg-gradient-to-r from-purple-500 to-pink-600 text-white py-5 px-6 rounded-2xl font-semibold flex items-center justify-center gap-3 shadow-lg shadow-purple-500/50 hover:scale-105">
                        <Camera className="w-6 h-6" />
                        Tomar Foto con Cámara
                      </div>
                    </button>
                    <div className="flex items-center gap-4">
                      <div className="flex-1 h-px bg-slate-700"></div>
                      <span className="text-slate-500 font-medium">o</span>
                      <div className="flex-1 h-px bg-slate-700"></div>
                    </div>
                    <input type="file" ref={fileInputRef} onChange={handleImageUpload} accept="image/*" className="hidden" />
                    <button onClick={() => fileInputRef.current?.click()} className="group/btn relative w-full overflow-hidden">
                      <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl blur-xl opacity-75 group-hover/btn:opacity-100 transition duration-300"></div>
                      <div className="relative bg-gradient-to-r from-blue-500 to-indigo-600 text-white py-5 px-6 rounded-2xl font-semibold flex items-center justify-center gap-3 shadow-lg shadow-blue-500/50 hover:scale-105">
                        <Upload className="w-6 h-6" />
                        Subir Imagen desde Dispositivo
                      </div>
                    </button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="rounded-2xl overflow-hidden border border-slate-700">
                      <img src={selectedImage} alt="Original" className="w-full h-80 object-contain bg-slate-900" />
                      <div className="bg-slate-800 p-3 text-center border-t border-slate-700">
                        <p className="text-sm text-slate-400 font-medium">Imagen Original</p>
                      </div>
                    </div>
                    <div className="flex gap-3">
                      {!isProcessing && !result && (
                        <button onClick={analyzeImage} className="flex-1 group/btn relative overflow-hidden">
                          <div className="absolute inset-0 bg-gradient-to-r from-green-600 to-emerald-600 rounded-xl blur-xl opacity-75 group-hover/btn:opacity-100 transition duration-300"></div>
                          <div className="relative bg-gradient-to-r from-green-500 to-emerald-600 text-white py-4 px-6 rounded-xl font-semibold flex items-center justify-center gap-2 shadow-lg shadow-green-500/50 hover:scale-105">
                            <Zap className="w-5 h-5" />
                            Analizar con IA {useTTA && '+ TTA'}
                          </div>
                        </button>
                      )}
                      <button onClick={resetTest} className="flex-1 bg-slate-700 border border-slate-600 text-slate-300 py-4 px-6 rounded-xl font-semibold hover:bg-slate-600 hover:border-slate-500 transition-all flex items-center justify-center gap-2">
                        <XCircle className="w-5 h-5" />
                        Nueva Prueba
                      </button>
                    </div>
                    {isProcessing && (
                      <div className="bg-blue-500/10 border border-blue-500/30 rounded-2xl p-6">
                        <div className="flex items-center gap-4 mb-4">
                          <Loader className="w-10 h-10 text-blue-400 animate-spin" />
                          <div>
                            <p className="font-bold text-blue-400 text-xl">Procesando...</p>
                            <p className="text-sm text-slate-400">{useTTA ? 'UNet++ B8 + TTA (6x)' : 'UNet++ B8'}</p>
                          </div>
                        </div>
                        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                          <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 animate-pulse rounded-full w-full"></div>
                        </div>
                      </div>
                    )}
                    {processedImage && result && result.success && (
                      <div className="rounded-2xl overflow-hidden border border-slate-700">
                        <img src={processedImage} alt="Procesada" className="w-full h-80 object-contain bg-slate-900" />
                        <div className="bg-slate-800 p-3 text-center border-t border-slate-700">
                          <p className="text-sm text-slate-400 font-medium">
                            Resultado Procesado
                            {result.procesamiento?.tta_usado && <span className="text-cyan-400"> • TTA {result.procesamiento.tta_transforms}x</span>}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* DERECHA */}
            <div className="space-y-6">
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-blue-500 to-indigo-600 p-2 rounded-xl">
                    <ImageIcon className="w-6 h-6 text-white" />
                  </div>
                  Tecnología IA v5.0
                </h3>
                <p className="text-slate-300">Sistema listo para pruebas en producción.</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Pruebas