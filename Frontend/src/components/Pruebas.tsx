/*  
import { useState, useRef, useEffect } from 'react'
import { Camera, Upload, Image as ImageIcon, Zap, CheckCircle, AlertCircle, Loader, XCircle, AlertTriangle, Settings, Wifi, Video, WifiOff, RefreshCw, Globe, Clock, HardDrive } from 'lucide-react'

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
  stream_url_local: string
  stream_url_proxy: string
  stream_url_public?: string
  tunnel_type?: string
  capabilities: string[]
  connected_at: string
  last_seen_ago: number
  has_photo?: boolean
  last_photo_time?: string
  last_photo_url?: string
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

  const API_URL = import.meta.env.VITE_API_URL || 'https://crackguard.angelproyect.com'

  // ═══════════════════════════════════════════════════════════════════════════
  // 🆕 CARGAR DISPOSITIVOS - COMPATIBLE CON BACKEND 4.3
  // ═══════════════════════════════════════════════════════════════════════════

  useEffect(() => {
    loadRaspberryDevices()
    const interval = setInterval(loadRaspberryDevices, 10000)
    return () => clearInterval(interval)
  }, [])

  // 🧹 CLEANUP: Liberar URLs de objetos al desmontar
  useEffect(() => {
    return () => {
      if (selectedImage && selectedImage.startsWith('blob:')) {
        URL.revokeObjectURL(selectedImage)
      }
      if (processedImage && processedImage.startsWith('blob:')) {
        URL.revokeObjectURL(processedImage)
      }
    }
  }, [selectedImage, processedImage])

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
  // AUTENTICACIÓN
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
  // 🆕 CAPTURAR FOTO - OPTIMIZADO + CACHE-BUSTING
  // ═══════════════════════════════════════════════════════════════════════════

  const captureAndAnalyzeFromRaspberry = async (deviceId: string) => {
    setIsCapturingFromRaspi(true)
    setError(null)
    setSelectedDevice(deviceId)

    try {
      console.log(`📸 Solicitando captura a ${deviceId}...`)

      // 1. Enviar comando de captura
      const cmdResponse = await fetch(`${API_URL}/api/rpi/capture/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution: '1920x1080', format: 'jpg' })
      })

      if (!cmdResponse.ok) {
        throw new Error('Error al enviar comando al Raspberry Pi')
      }

      console.log('✅ Comando enviado, esperando foto...')
      
      // 2. Esperar a que RPi procese y envíe la foto
      await new Promise(resolve => setTimeout(resolve, 5000))

      // 3. 🆕 CACHE-BUSTING: Agregar timestamp para forzar recarga
      const timestamp = Date.now()
      const photoUrl = `${API_URL}/api/rpi/latest-photo/${deviceId}?t=${timestamp}`
      console.log('📥 Descargando foto desde:', photoUrl)

      const photoResponse = await fetch(photoUrl, {
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      })

      if (!photoResponse.ok) {
        throw new Error('No se pudo obtener la foto del Raspberry Pi')
      }

      // 4. Convertir blob de imagen a File
      const blob = await photoResponse.blob()
      const file = new File([blob], `raspberry_${deviceId}_${timestamp}.jpg`, { type: 'image/jpeg' })

      // 5. ✅ Revocar URL anterior para liberar memoria
      if (selectedImage && selectedImage.startsWith('blob:')) {
        URL.revokeObjectURL(selectedImage)
      }

      setSelectedFile(file)
      setSelectedImage(URL.createObjectURL(blob))
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
  // 🆕 VER ÚLTIMA FOTO - CON CACHE-BUSTING
  // ═══════════════════════════════════════════════════════════════════════════

  const viewLastPhoto = async (deviceId: string) => {
    try {
      const timestamp = Date.now()
      const photoUrl = `${API_URL}/api/rpi/latest-photo/${deviceId}?t=${timestamp}`
      
      const response = await fetch(photoUrl, {
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      })

      if (!response.ok) {
        setError('No hay foto guardada para este dispositivo')
        return
      }

      const blob = await response.blob()
      const file = new File([blob], `raspberry_${deviceId}_${timestamp}.jpg`, { type: 'image/jpeg' })

      if (selectedImage && selectedImage.startsWith('blob:')) {
        URL.revokeObjectURL(selectedImage)
      }

      setSelectedFile(file)
      setSelectedImage(URL.createObjectURL(blob))
      setResult(null)
      setProcessedImage(null)
      setSelectedDevice(deviceId)

      console.log('✅ Última foto cargada con timestamp:', timestamp)
    } catch (err) {
      setError('Error al cargar la última foto')
      console.error('❌ Error:', err)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // STREAMING
  // ═══════════════════════════════════════════════════════════════════════════

  const startStreaming = async (deviceId: string) => {
    setIsStartingStream(true)
    setError(null)
    setSelectedDevice(deviceId)

    try {
      console.log(`🎬 Iniciando streaming en ${deviceId}...`)

      const response = await fetch(`${API_URL}/api/rpi/streaming/start/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) {
        throw new Error('Error al iniciar streaming')
      }

      console.log('✅ Comando de inicio enviado')
      await new Promise(resolve => setTimeout(resolve, 8000))
      await loadRaspberryDevices()

      const updatedDevice = raspberryDevices.find(d => d.device_id === deviceId)

      if (!updatedDevice) {
        throw new Error('Dispositivo no encontrado después de actualizar')
      }

      let finalStreamUrl: string

      if (updatedDevice.stream_url_public) {
        finalStreamUrl = updatedDevice.stream_url_public
        console.log('✅ Usando Cloudflare Tunnel:', finalStreamUrl)
      } else {
        finalStreamUrl = `${API_URL}/api/stream/${deviceId}`
        console.log('⚠️ Cloudflare no disponible, usando proxy backend')
      }

      setStreamUrl(finalStreamUrl)
      setIsStreamActive(true)

      console.log('✅ Streaming activo:', finalStreamUrl)

    } catch (err) {
      console.error('❌ Error al iniciar streaming:', err)
      setError('No se pudo iniciar el streaming')
    } finally {
      setIsStartingStream(false)
    }
  }

  const stopStreaming = async (deviceId: string) => {
    setIsStoppingStream(true)
    setError(null)

    try {
      console.log(`🛑 Deteniendo streaming en ${deviceId}...`)

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

      await loadRaspberryDevices()

    } catch (err) {
      console.error('❌ Error al detener streaming:', err)
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
        if (selectedImage && selectedImage.startsWith('blob:')) {
          URL.revokeObjectURL(selectedImage)
        }
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
        if (selectedImage && selectedImage.startsWith('blob:')) {
          URL.revokeObjectURL(selectedImage)
        }
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
    if (selectedImage && selectedImage.startsWith('blob:')) {
      URL.revokeObjectURL(selectedImage)
    }
    if (processedImage && processedImage.startsWith('blob:')) {
      URL.revokeObjectURL(processedImage)
    }
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
  // UTILIDADES
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
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">PRUEBAS v5.1 + BACKEND 4.3 + CACHE-BUSTING</span>
            </div>

            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">
              Prueba el Sistema
            </h2>

            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              UNet++ EfficientNet-B8 + TTA + Análisis Morfológico + Raspberry Pi + Cloudflare Tunnel
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
                      Dispositivos Raspberry Pi + Cloudflare Tunnel
                      <span className="text-xs bg-green-500/20 text-green-400 px-3 py-1 rounded-full border border-green-500/30">
                        Backend 4.3 + Cache-Busting
                      </span>
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
                              
                              {device.stream_url_public && (
                                <p className="text-xs text-cyan-400 mt-1 flex items-center gap-1">
                                  <Globe className="w-3 h-3" />
                                  Cloudflare Tunnel activo
                                  {device.tunnel_type && (
                                    <span className="text-slate-500">• {device.tunnel_type}</span>
                                  )}
                                </p>
                              )}
                              
                              {device.has_photo && (
                                <p className="text-xs text-green-400 mt-1 flex items-center gap-1">
                                  <HardDrive className="w-3 h-3" />
                                  📸 Última foto guardada: {new Date(device.last_photo_time || '').toLocaleTimeString('es-PE')}
                                </p>
                              )}
                              <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
                                <Clock className="w-3 h-3" />
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
                              <span
                                key={idx}
                                className="bg-purple-500/20 text-purple-400 text-xs font-semibold px-3 py-1 rounded-full border border-purple-500/30"
                              >
                                {cap}
                              </span>
                            ))}
                          </div>

                          <div className="grid grid-cols-3 gap-3">
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
                              Capturar
                            </button>

                            {device.has_photo && (
                              <button
                                onClick={() => viewLastPhoto(device.device_id)}
                                className="bg-gradient-to-r from-emerald-500 to-green-600 text-white py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all flex items-center justify-center gap-2"
                              >
                                <HardDrive className="w-5 h-5" />
                                Ver Última
                              </button>
                            )}

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
                                    Detener
                                  </>
                                ) : (
                                  <>
                                    <Video className="w-5 h-5" />
                                    Stream
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
                                onLoad={() => console.log('✅ Iframe cargado:', streamUrl)}
                                onError={(e) => {
                                  console.error('❌ Error al cargar streaming iframe:', e)
                                  setError('No se pudo conectar al streaming. Reintentando...')
                                  setTimeout(() => {
                                    loadRaspberryDevices()
                                  }, 3000)
                                }}
                              />
                              <div className="bg-slate-900 p-3 text-center">
                                <p className="text-sm text-purple-400 font-semibold flex items-center justify-center gap-2">
                                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                                  STREAMING EN VIVO • WebRTC • MediaMTX
                                  {device.stream_url_public && (
                                    <span className="text-cyan-400 ml-2 flex items-center gap-1">
                                      <Globe className="w-3 h-3" />
                                      Cloudflare Tunnel
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
            {/* COLUMNA IZQUIERDA: CAPTURA *//*}
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
                        <AlertCircle className="w-6 h-6" />
                        Instrucciones
                      </h4>
                      <ul className="text-sm text-slate-300 space-y-3">
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Captura o sube una imagen de estructura de concreto</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Usa Raspberry Pi con Cloudflare Tunnel para streaming global</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>🆕 Backend 4.3: fotos guardadas en disco con cache-busting</span>
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
                  </div>
                )}
              </div>
            </div>

            {/* COLUMNA DERECHA: RESULTADOS *//*}
            <div className="space-y-6">
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-blue-500 to-indigo-600 p-2 rounded-xl">
                    <ImageIcon className="w-6 h-6 text-white" />
                  </div>
                  {result ? 'Resultados del Análisis' : 'Tecnología IA v5.1'}
                </h3>

                {!result ? (
                  <div className="space-y-6">
                    <div className="bg-slate-900/50 border border-slate-700 rounded-2xl p-6">
                      <h4 className="font-bold text-cyan-400 mb-4 text-lg">🧠 Arquitectura</h4>
                      <ul className="space-y-3 text-slate-300 text-sm">
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                          <span><strong>UNet++</strong> con EfficientNet-B8 encoder</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                          <span><strong>TTA opcional</strong> con 6 transformaciones</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                          <span><strong>Análisis morfológico</strong> de patrones de grietas</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                          <span><strong>Detección de causa probable</strong> y severidad</span>
                        </li>
                      </ul>
                    </div>

                    <div className="bg-slate-900/50 border border-slate-700 rounded-2xl p-6">
                      <h4 className="font-bold text-purple-400 mb-4 text-lg">📡 Raspberry Pi + Cloudflare</h4>
                      <ul className="space-y-3 text-slate-300 text-sm">
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                          <span><strong>Captura remota</strong> desde Raspberry Pi Camera</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                          <span><strong>Streaming WebRTC</strong> en tiempo real con MediaMTX</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                          <span><strong>Cloudflare Tunnel</strong> para acceso global seguro</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                          <span><strong>🆕 Cache-busting</strong> - siempre muestra foto más reciente</span>
                        </li>
                      </ul>
                    </div>

                    <div className="bg-green-500/10 border border-green-500/30 rounded-2xl p-6">
                      <h4 className="font-bold text-green-400 mb-3 text-lg">🆕 Backend 4.3 + Cache-Busting</h4>
                      <ul className="space-y-2 text-slate-300 text-sm">
                        <li>• Fotos guardadas en <code className="bg-slate-800 px-2 py-1 rounded text-cyan-400">/app/uploads</code></li>
                        <li>• ✅ Cache-busting con timestamp en URL</li>
                        <li>• ✅ Headers no-cache para navegador</li>
                        <li>• ✅ Limpieza automática de Blob URLs</li>
                        <li>• Metadata ligera en RAM (JSON)</li>
                        <li>• Limpieza automática cada 1 hora</li>
                        <li>• Máx 5 fotos por dispositivo • Retención: 24h</li>
                      </ul>
                    </div>

                    <div className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/30 rounded-2xl p-6">
                      <h4 className="font-bold text-blue-400 mb-3 text-lg">💡 Casos de Uso</h4>
                      <ul className="space-y-2 text-slate-300 text-sm">
                        <li>• Inspección de puentes y viaductos</li>
                        <li>• Monitoreo de edificios y estructuras</li>
                        <li>• Evaluación de daños post-desastre</li>
                        <li>• Mantenimiento preventivo de infraestructura</li>
                        <li>• Inspección remota con Raspberry Pi</li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className={`${getSeveridadBg(result.metricas.severidad)} border rounded-2xl p-6`}>
                      <div className="flex items-center gap-3 mb-4">
                        <span className="text-4xl">{getSeveridadIcon(result.metricas.severidad)}</span>
                        <div>
                          <p className="text-sm text-slate-400 font-medium">Severidad</p>
                          <p className={`text-2xl font-black ${getSeveridadColor(result.metricas.severidad)}`}>
                            {result.metricas.severidad.toUpperCase()}
                          </p>
                        </div>
                      </div>
                      <p className="text-sm text-slate-300">
                        Estado: <span className="font-semibold">{result.metricas.estado}</span>
                      </p>
                      <p className="text-sm text-slate-300 mt-1">
                        Confianza: <span className="font-semibold">{result.metricas.confianza.toFixed(1)}%</span>
                      </p>
                    </div>

                    <div className="bg-slate-900/50 border border-slate-700 rounded-2xl p-6">
                      <h4 className="font-bold text-white mb-4 text-lg">📊 Métricas de Detección</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs text-slate-500 mb-1">Grietas Detectadas</p>
                          <p className="text-2xl font-bold text-cyan-400">{result.metricas.num_grietas_detectadas}</p>
                        </div>
                        <div>
                          <p className="text-xs text-slate-500 mb-1">Área Afectada</p>
                                                    <p className="text-2xl font-bold text-orange-400">{result.metricas.porcentaje_grietas.toFixed(2)}%</p>
                        </div>
                        {result.metricas.longitud_total_px && (
                          <div>
                            <p className="text-xs text-slate-500 mb-1">Longitud Total</p>
                            <p className="text-lg font-bold text-blue-400">{result.metricas.longitud_total_px.toFixed(0)} px</p>
                          </div>
                        )}
                        {result.metricas.ancho_promedio_px && (
                          <div>
                            <p className="text-xs text-slate-500 mb-1">Ancho Promedio</p>
                            <p className="text-lg font-bold text-purple-400">{result.metricas.ancho_promedio_px.toFixed(1)} px</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {result.metricas.analisis_morfologico && (
                      <div className="bg-gradient-to-br from-orange-500/10 to-red-500/10 border border-orange-500/30 rounded-2xl p-6">
                        <h4 className="font-bold text-orange-400 mb-4 text-lg flex items-center gap-2">
                          {getPatronIcon(result.metricas.analisis_morfologico.patron_general)}
                          Análisis Morfológico
                        </h4>
                        
                        <div className="space-y-4">
                          <div>
                            <p className="text-sm text-slate-400 mb-1">Patrón Detectado</p>
                            <p className="text-lg font-bold text-white">{result.metricas.analisis_morfologico.patron_general.replace('_', ' ').toUpperCase()}</p>
                            <p className="text-sm text-slate-300 mt-2">{result.metricas.analisis_morfologico.descripcion_patron}</p>
                          </div>

                          <div className="bg-slate-900/50 border border-orange-500/20 rounded-xl p-4">
                            <p className="text-sm text-orange-400 font-semibold mb-2">⚠️ Causa Probable</p>
                            <p className="text-sm text-slate-300">{result.metricas.analisis_morfologico.causa_probable}</p>
                          </div>

                          <div className="bg-slate-900/50 border border-cyan-500/20 rounded-xl p-4">
                            <p className="text-sm text-cyan-400 font-semibold mb-2">💡 Recomendación</p>
                            <p className="text-sm text-slate-300">{result.metricas.analisis_morfologico.recomendacion}</p>
                          </div>

                          <div>
                            <p className="text-sm text-slate-400 mb-3">Distribución de Orientaciones</p>
                            <div className="grid grid-cols-2 gap-2">
                              {Object.entries(result.metricas.analisis_morfologico.distribucion_orientaciones).map(([key, val]) => (
                                <div key={key} className={`${getOrientacionColor(key)} border px-3 py-2 rounded-lg text-center`}>
                                  <p className="text-xs font-semibold uppercase">{key}</p>
                                  <p className="text-lg font-bold">{val}</p>
                                </div>
                              ))}
                            </div>
                          </div>

                          {result.metricas.analisis_morfologico.grietas_principales.length > 0 && (
                            <div>
                              <p className="text-sm text-slate-400 mb-3">Grietas Principales ({result.metricas.analisis_morfologico.grietas_principales.length})</p>
                              <div className="space-y-2 max-h-48 overflow-y-auto">
                                {result.metricas.analisis_morfologico.grietas_principales.slice(0, 5).map((grieta) => (
                                  <div key={grieta.id} className="bg-slate-900/70 border border-slate-700 rounded-lg p-3 text-xs">
                                    <div className="flex items-center justify-between mb-2">
                                      <span className={`${getOrientacionColor(grieta.orientacion)} px-2 py-1 rounded-md font-semibold text-xs border`}>
                                        {grieta.orientacion.toUpperCase()}
                                      </span>
                                      <span className="text-slate-400">Grieta #{grieta.id}</span>
                                    </div>
                                    <div className="grid grid-cols-2 gap-2 text-slate-300">
                                      <div>
                                        <span className="text-slate-500">Longitud:</span> {safeToFixed(grieta.longitud_px, 0)} px
                                      </div>
                                      <div>
                                        <span className="text-slate-500">Ancho:</span> {safeToFixed(grieta.ancho_promedio_px, 1)} px
                                      </div>
                                      {grieta.angulo_grados !== null && (
                                        <div>
                                          <span className="text-slate-500">Ángulo:</span> {safeToFixed(grieta.angulo_grados, 0)}°
                                        </div>
                                      )}
                                      <div>
                                        <span className="text-slate-500">Área:</span> {safeToFixed(grieta.area_px, 0)} px²
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {result.procesamiento && (
                      <div className="bg-slate-900/50 border border-slate-700 rounded-2xl p-6">
                        <h4 className="font-bold text-slate-300 mb-4 text-lg">⚙️ Información de Procesamiento</h4>
                        <div className="space-y-2 text-sm text-slate-400">
                          <p><span className="text-slate-500">Arquitectura:</span> <span className="text-white font-semibold">{result.procesamiento.architecture}</span></p>
                          <p><span className="text-slate-500">Encoder:</span> <span className="text-white font-semibold">{result.procesamiento.encoder}</span></p>
                          <p><span className="text-slate-500">TTA:</span> <span className={result.procesamiento.tta_usado ? 'text-cyan-400' : 'text-slate-400'}>
                            {result.procesamiento.tta_usado ? `✓ Activado (${result.procesamiento.tta_transforms}x)` : '✗ Desactivado'}
                          </span></p>
                          <p><span className="text-slate-500">Threshold:</span> <span className="text-white">{result.procesamiento.threshold}</span></p>
                          {result.procesamiento.cpu_optimized && (
                            <p><span className="text-slate-500">Optimización CPU:</span> <span className="text-green-400">✓ Activado ({result.procesamiento.cpu_threads} threads)</span></p>
                          )}
                          {result.procesamiento.original_dimensions && (
                            <p><span className="text-slate-500">Dimensiones originales:</span> <span className="text-white">
                              {result.procesamiento.original_dimensions.width} × {result.procesamiento.original_dimensions.height} px
                            </span></p>
                          )}
                        </div>
                      </div>
                    )}

                    <div className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-2xl p-6">
                      <h4 className="font-bold text-green-400 mb-3 text-lg">✅ Análisis Completado</h4>
                      <p className="text-sm text-slate-300 mb-3">
                        El sistema ha detectado y clasificado las grietas en la estructura. 
                        Revisa las recomendaciones y considera una inspección profesional si es necesario.
                      </p>
                      <p className="text-xs text-slate-500">
                        Timestamp: {new Date(result.timestamp).toLocaleString('es-PE')}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Pruebas */



import { useState, useRef, useEffect } from 'react'
import { Camera, Upload, Image as ImageIcon, Zap, CheckCircle, AlertCircle, Loader, XCircle, AlertTriangle, Settings, Wifi, Video, WifiOff, RefreshCw, Globe, Clock, HardDrive, Zap as ZapIcon } from 'lucide-react'

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
  stream_url_local: string
  stream_url_proxy?: string
  stream_url_public?: string
  tunnel_type?: string
  capabilities: string[]
  connected_at: string
  last_seen_ago: number
  has_photo?: boolean
  last_photo_time?: string
  last_photo_url?: string
  last_photo_size_kb?: number
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

  const API_URL = import.meta.env.VITE_API_URL || 'https://crackguard.angelproyect.com'

  // ═══════════════════════════════════════════════════════════════════════════
  // 🆕 CARGAR DISPOSITIVOS - COMPATIBLE CON BACKEND 5.0
  // ═══════════════════════════════════════════════════════════════════════════

  useEffect(() => {
    loadRaspberryDevices()
    const interval = setInterval(loadRaspberryDevices, 10000)
    return () => clearInterval(interval)
  }, [])

  // 🧹 CLEANUP: Liberar URLs de objetos al desmontar
  useEffect(() => {
    return () => {
      if (selectedImage && selectedImage.startsWith('blob:')) {
        URL.revokeObjectURL(selectedImage)
      }
      if (processedImage && processedImage.startsWith('blob:')) {
        URL.revokeObjectURL(processedImage)
      }
    }
  }, [selectedImage, processedImage])

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
  // AUTENTICACIÓN
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
  // 🚀 CAPTURAR FOTO - OPTIMIZADO CON CACHE-BUSTING
  // ═══════════════════════════════════════════════════════════════════════════

  const captureAndAnalyzeFromRaspberry = async (deviceId: string) => {
    setIsCapturingFromRaspi(true)
    setError(null)
    setSelectedDevice(deviceId)

    try {
      console.log(`📸 Solicitando captura a ${deviceId}... `)

      // 1. Enviar comando de captura
      const cmdResponse = await fetch(`${API_URL}/api/rpi/capture/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution: '1920x1080', format: 'jpg' })
      })

      if (!cmdResponse.ok) {
        throw new Error('Error al enviar comando al Raspberry Pi')
      }

      console.log('✅ Comando enviado, esperando foto...')
      
      // 2.  Esperar a que RPi procese y envíe la foto (multipart es más rápido)
      await new Promise(resolve => setTimeout(resolve, 3000)) // 3s en vez de 5s

      // 3. 🚀 CACHE-BUSTING: Agregar timestamp para forzar recarga
      const timestamp = Date.now()
      const photoUrl = `${API_URL}/api/rpi/latest-photo/${deviceId}?t=${timestamp}`
      console.log('📥 Descargando foto desde:', photoUrl)

      const photoResponse = await fetch(photoUrl, {
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      })

      if (!photoResponse.ok) {
        throw new Error('No se pudo obtener la foto del Raspberry Pi')
      }

      // 4.  Convertir blob de imagen a File
      const blob = await photoResponse.blob()
      const file = new File([blob], `raspberry_${deviceId}_${timestamp}.jpg`, { type: 'image/jpeg' })

      // 5. ✅ Revocar URL anterior para liberar memoria
      if (selectedImage && selectedImage.startsWith('blob:')) {
        URL.revokeObjectURL(selectedImage)
      }

      setSelectedFile(file)
      setSelectedImage(URL.createObjectURL(blob))
      setResult(null)
      setProcessedImage(null)

      console.log('✅ Foto lista para análisis (multipart)')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al capturar desde Raspberry Pi')
      console. error('❌ Error:', err)
    } finally {
      setIsCapturingFromRaspi(false)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // 🆕 VER ÚLTIMA FOTO - CON CACHE-BUSTING
  // ═══════════════════════════════════════════════════════════════════════════

  const viewLastPhoto = async (deviceId: string) => {
    try {
      const timestamp = Date.now()
      const photoUrl = `${API_URL}/api/rpi/latest-photo/${deviceId}?t=${timestamp}`
      
      const response = await fetch(photoUrl, {
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      })

      if (!response. ok) {
        setError('No hay foto guardada para este dispositivo')
        return
      }

      const blob = await response. blob()
      const file = new File([blob], `raspberry_${deviceId}_${timestamp}.jpg`, { type: 'image/jpeg' })

      if (selectedImage && selectedImage.startsWith('blob:')) {
        URL.revokeObjectURL(selectedImage)
      }

      setSelectedFile(file)
      setSelectedImage(URL. createObjectURL(blob))
      setResult(null)
      setProcessedImage(null)
      setSelectedDevice(deviceId)

      console.log('✅ Última foto cargada con timestamp:', timestamp)
    } catch (err) {
      setError('Error al cargar la última foto')
      console.error('❌ Error:', err)
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // STREAMING
  // ═══════════════════════════════════════════════════════════════════════════

  const startStreaming = async (deviceId: string) => {
    setIsStartingStream(true)
    setError(null)
    setSelectedDevice(deviceId)

    try {
      console.log(`🎬 Iniciando streaming en ${deviceId}...`)

      const response = await fetch(`${API_URL}/api/rpi/streaming/start/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) {
        throw new Error('Error al iniciar streaming')
      }

      console.log('✅ Comando de inicio enviado')
      await new Promise(resolve => setTimeout(resolve, 10000)) // Esperar 10s
      await loadRaspberryDevices()

      const updatedDevice = raspberryDevices.find(d => d.device_id === deviceId)

      if (!updatedDevice) {
        throw new Error('Dispositivo no encontrado después de actualizar')
      }

      let finalStreamUrl: string

      // Prioridad: URL pública de FRP > URL local
      if (updatedDevice.stream_url_public) {
        finalStreamUrl = updatedDevice. stream_url_public
        console.log('✅ Usando FRP Tunnel:', finalStreamUrl)
      } else if (updatedDevice.stream_url_local) {
        finalStreamUrl = updatedDevice.stream_url_local
        console.log('⚠️ Usando URL local:', finalStreamUrl)
      } else {
        throw new Error('No hay URL de streaming disponible')
      }

      setStreamUrl(finalStreamUrl)
      setIsStreamActive(true)

      console.log('✅ Streaming activo:', finalStreamUrl)

    } catch (err) {
      console.error('❌ Error al iniciar streaming:', err)
      setError('No se pudo iniciar el streaming')
    } finally {
      setIsStartingStream(false)
    }
  }

  const stopStreaming = async (deviceId: string) => {
    setIsStoppingStream(true)
    setError(null)

    try {
      console.log(`🛑 Deteniendo streaming en ${deviceId}...`)

      const response = await fetch(`${API_URL}/api/rpi/streaming/stop/${deviceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response. ok) {
        throw new Error('Error al detener streaming')
      }

      console.log('✅ Comando de detención enviado')

      setStreamUrl(null)
      setIsStreamActive(false)
      setSelectedDevice(null)

      await loadRaspberryDevices()

    } catch (err) {
      console.error('❌ Error al detener streaming:', err)
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
    if (file) {
      if (file.size > 50 * 1024 * 1024) { // 50MB
        setError('El archivo es demasiado grande.  Máximo 50MB.')
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
        if (selectedImage && selectedImage.startsWith('blob:')) {
          URL. revokeObjectURL(selectedImage)
        }
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
      setError('No se pudo acceder a la cámara.  Verifica los permisos.')
    }
  }

  const capturePhoto = () => {
    if (! videoRef.current || !canvasRef.current) return
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
        if (selectedImage && selectedImage.startsWith('blob:')) {
          URL.revokeObjectURL(selectedImage)
        }
        setSelectedImage(URL.createObjectURL(blob))
        setResult(null)
        setProcessedImage(null)
        closeCamera()
      }
    }, 'image/jpeg', 0.95)
  }

  const closeCamera = () => {
    if (stream) {
      stream. getTracks().forEach(track => track.stop())
      setStream(null)
    }
    setIsCameraOpen(false)
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // 🚀 ANALIZAR IMAGEN - CON MULTIPART
  // ═══════════════════════════════════════════════════════════════════════════

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
      // 🚀 USAR MULTIPART EN VEZ DE JSON
      const formData = new FormData()
      formData.append('image', selectedFile)
      formData.append('use_tta', useTTA. toString())

      console.log('📤 Enviando imagen con multipart/form-data...')

      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData, // Sin Content-Type (se agrega automáticamente)
      })

      if (!response.ok) {
        const contentType = response.headers.get('content-type')
        if (contentType?. includes('application/json')) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Error en la predicción')
        } else {
          throw new Error(`Error del servidor: ${response.status}`)
        }
      }

      const data: PredictResponse = await response.json()

      if (! data.success) {
        throw new Error(data.error || 'Error en la predicción')
      }

      setResult(data)
      if (data.imagen_overlay) {
        setProcessedImage(data.imagen_overlay)
      }

      console.log('✅ Análisis completado')
    } catch (err) {
      setError(err instanceof Error ? err. message : 'Error desconocido al analizar la imagen')
    } finally {
      setIsProcessing(false)
    }
  }

  const resetTest = () => {
    if (selectedImage && selectedImage.startsWith('blob:')) {
      URL.revokeObjectURL(selectedImage)
    }
    if (processedImage && processedImage.startsWith('blob:')) {
      URL.revokeObjectURL(processedImage)
    }
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
  // UTILIDADES
  // ═══════════════════════════════════════════════════════════════════════════

  const getSeveridadColor = (severidad: string) => {
    switch (severidad. toLowerCase()) {
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
    switch (severidad. toLowerCase()) {
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
    switch (severidad. toLowerCase()) {
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
    return value !== undefined && value !== null ? value. toFixed(decimals) : '0. 00'
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
              <ZapIcon className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">FRONTEND v5.0 + BACKEND v5.0 MULTIPART 🚀</span>
            </div>

            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">
              Prueba el Sistema
            </h2>

            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              UNet++ EfficientNet-B8 + TTA + Análisis Morfológico + Raspberry Pi + FRP Tunnel + Multipart (3× más rápido)
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
                  setShowRaspberryPanel(! showRaspberryPanel)
                  if (! showRaspberryPanel) loadRaspberryDevices()
                }}
                className="inline-flex items-center gap-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white px-6 py-3 rounded-full font-semibold hover:scale-105 transition-all shadow-lg shadow-purple-500/50"
              >
                {raspberryDevices.length > 0 ?  (
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
                      Dispositivos Raspberry Pi + FRP Tunnel
                      <span className="text-xs bg-green-500/20 text-green-400 px-3 py-1 rounded-full border border-green-500/30">
                        Backend v5.0 Multipart 🚀
                      </span>
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
                        Inicia el cliente bash v5.0 en tu Raspberry Pi
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
                              
                              {device.stream_url_public && (
                                <p className="text-xs text-cyan-400 mt-1 flex items-center gap-1">
                                  <Globe className="w-3 h-3" />
                                  FRP Tunnel activo ({device.tunnel_type || 'frp'})
                                </p>
                              )}
                              
                              {device.has_photo && (
                                <p className="text-xs text-green-400 mt-1 flex items-center gap-1">
                                  <HardDrive className="w-3 h-3" />
                                  📸 Última foto: {new Date(device.last_photo_time || '').toLocaleTimeString('es-PE')}
                                  {device.last_photo_size_kb && ` • ${device.last_photo_size_kb}KB`}
                                </p>
                              )}
                              <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
                                <Clock className="w-3 h-3" />
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
                              <span
                                key={idx}
                                className="bg-purple-500/20 text-purple-400 text-xs font-semibold px-3 py-1 rounded-full border border-purple-500/30"
                              >
                                {cap}
                              </span>
                            ))}
                          </div>

                          <div className="grid grid-cols-3 gap-3">
                            <button
                              onClick={() => handleAuthForAction(device.device_id, 'capture')}
                              disabled={isCapturingFromRaspi}
                              className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                            >
                              {isCapturingFromRaspi && selectedDevice === device.device_id ?  (
                                <Loader className="w-5 h-5 animate-spin" />
                              ) : (
                                <Camera className="w-5 h-5" />
                              )}
                              Capturar
                            </button>

                            {device.has_photo && (
                              <button
                                onClick={() => viewLastPhoto(device.device_id)}
                                className="bg-gradient-to-r from-emerald-500 to-green-600 text-white py-3 px-4 rounded-xl font-semibold hover:scale-105 transition-all flex items-center justify-center gap-2"
                              >
                                <HardDrive className="w-5 h-5" />
                                Ver Última
                              </button>
                            )}

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
                                {(isStartingStream || isStoppingStream) && selectedDevice === device. device_id ? (
                                  <Loader className="w-5 h-5 animate-spin" />
                                ) : device.streaming_active || (isStreamActive && selectedDevice === device.device_id) ? (
                                  <>
                                    <XCircle className="w-5 h-5" />
                                    Detener
                                  </>
                                ) : (
                                  <>
                                    <Video className="w-5 h-5" />
                                    Stream
                                  </>
                                )}
                              </button>
                            )}
                          </div>

                          {/* IFRAME STREAMING */}
                          {isStreamActive && streamUrl && selectedDevice === device.device_id && (
                            <div className="mt-4 bg-black rounded-xl overflow-hidden border border-purple-500/50">
                              <iframe
                                src={streamUrl}
                                className="w-full aspect-video"
                                allow="camera; microphone"
                                title={`Stream ${device.device_id}`}
                                onLoad={() => console.log('✅ Iframe cargado:', streamUrl)}
                                onError={(e) => {
                                  console.error('❌ Error al cargar streaming iframe:', e)
                                  setError('No se pudo conectar al streaming.  Reintentando...')
                                  setTimeout(() => {
                                    loadRaspberryDevices()
                                  }, 3000)
                                }}
                              />
                              <div className="bg-slate-900 p-3 text-center">
                                <p className="text-sm text-purple-400 font-semibold flex items-center justify-center gap-2">
                                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                                  STREAMING EN VIVO • WebRTC • MediaMTX
                                  {device.stream_url_public && (
                                    <span className="text-cyan-400 ml-2 flex items-center gap-1">
                                      <Globe className="w-3 h-3" />
                                      FRP Tunnel
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

            {/* MODAL PASSWORD */}
            {showPasswordModal && (
              <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
                <div className="bg-slate-900 rounded-2xl p-8 max-w-md w-full border border-slate-700">
                  <h3 className="text-xl font-bold text-white mb-4">Acceso a Raspberry Pi</h3>
                  <p className="text-slate-400 mb-6">Ingresa la contraseña para acceder a la cámara/streaming</p>

                  <input
                    type="password"
                    value={passwordInput}
                    onChange={(e) => setPasswordInput(e. target.value)}
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

          {/* ALERT ERROR */}
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
            {/* COLUMNA IZQUIERDA: CAPTURA */}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 rounded-3xl blur-2xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8 hover:border-cyan-500/50 transition-all duration-300">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-2 rounded-xl">
                    <Camera className="w-6 h-6 text-white" />
                  </div>
                  Captura de Imagen
                </h3>

                {/* MODAL CÁMARA LOCAL */}
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

                {! selectedImage ?  (
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
                        <AlertCircle className="w-6 h-6" />
                        Instrucciones
                      </h4>
                      <ul className="text-sm text-slate-300 space-y-3">
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Captura o sube una imagen de estructura de concreto</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>Usa Raspberry Pi con FRP Tunnel para acceso global</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>🚀 Backend v5.0: Multipart (3× más rápido que base64)</span>
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
                      {! isProcessing && ! result && (
                        <button
                          onClick={analyzeImage}
                          disabled={! selectedFile}
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
                            <p className="text-xs text-green-400 mt-1">🚀 Multipart (3× más rápido)</p>
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
                            {result.procesamiento?. tta_usado && <span className="text-cyan-400"> • TTA {result.procesamiento.tta_transforms}x</span>}
                            {result.procesamiento?.cpu_optimized && <span className="text-green-400"> • CPU Optimizado</span>}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
                        {/* COLUMNA DERECHA: RESULTADOS */}
            <div className="space-y-6">
              <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-6 md:p-8">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <div className="bg-gradient-to-br from-blue-500 to-indigo-600 p-2 rounded-xl">
                    <ImageIcon className="w-6 h-6 text-white" />
                  </div>
                  {result ?  'Resultados del Análisis' : 'Tecnología IA v5.0 🚀'}
                </h3>

                {! result ? (
                  <div className="space-y-6">
                    <div className="bg-slate-900/50 border border-slate-700 rounded-2xl p-6">
                      <h4 className="font-bold text-cyan-400 mb-4 text-lg">🧠 Arquitectura</h4>
                      <ul className="space-y-3 text-slate-300 text-sm">
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                          <span><strong>UNet++</strong> con EfficientNet-B8 encoder</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                          <span><strong>TTA opcional</strong> con 6 transformaciones</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                          <span><strong>Análisis morfológico</strong> de patrones de grietas</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                          <span><strong>Detección de causa probable</strong> y severidad</span>
                        </li>
                      </ul>
                    </div>

                    <div className="bg-slate-900/50 border border-slate-700 rounded-2xl p-6">
                      <h4 className="font-bold text-purple-400 mb-4 text-lg">📡 Raspberry Pi + FRP</h4>
                      <ul className="space-y-3 text-slate-300 text-sm">
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                          <span><strong>Captura remota</strong> desde Raspberry Pi Camera</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                          <span><strong>Streaming WebRTC</strong> en tiempo real con MediaMTX</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                          <span><strong>FRP Tunnel</strong> para acceso global seguro</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                          <span><strong>🚀 Multipart</strong> - 3× más rápido que base64</span>
                        </li>
                      </ul>
                    </div>

                    <div className="bg-green-500/10 border border-green-500/30 rounded-2xl p-6">
                      <h4 className="font-bold text-green-400 mb-3 text-lg">🚀 Backend v5.0 Multipart</h4>
                      <ul className="space-y-2 text-slate-300 text-sm">
                        <li>• ✅ Subida con <code className="bg-slate-800 px-2 py-1 rounded text-cyan-400">multipart/form-data</code></li>
                        <li>• ✅ 3× más rápido que base64 (0% overhead)</li>
                        <li>• ✅ Soporta archivos hasta 50MB</li>
                        <li>• ✅ Cache-busting con timestamp en URL</li>
                        <li>• ✅ Headers no-cache para navegador</li>
                        <li>• ✅ Limpieza automática de Blob URLs</li>
                        <li>• Máx 5 fotos por dispositivo • Retención: 24h</li>
                      </ul>
                    </div>

                    <div className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/30 rounded-2xl p-6">
                      <h4 className="font-bold text-blue-400 mb-3 text-lg">💡 Casos de Uso</h4>
                      <ul className="space-y-2 text-slate-300 text-sm">
                        <li>• Inspección de puentes y viaductos</li>
                        <li>• Monitoreo de edificios y estructuras</li>
                        <li>• Evaluación de daños post-desastre</li>
                        <li>• Mantenimiento preventivo de infraestructura</li>
                        <li>• Inspección remota con Raspberry Pi</li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className={`${getSeveridadBg(result.metricas. severidad)} border rounded-2xl p-6`}>
                      <div className="flex items-center gap-3 mb-4">
                        <span className="text-4xl">{getSeveridadIcon(result.metricas.severidad)}</span>
                        <div>
                          <p className="text-sm text-slate-400 font-medium">Severidad</p>
                          <p className={`text-2xl font-black ${getSeveridadColor(result.metricas.severidad)}`}>
                            {result.metricas.severidad. toUpperCase()}
                          </p>
                        </div>
                      </div>
                      <p className="text-sm text-slate-300">
                        Estado: <span className="font-semibold">{result.metricas. estado}</span>
                      </p>
                      <p className="text-sm text-slate-300 mt-1">
                        Confianza: <span className="font-semibold">{result.metricas.confianza. toFixed(1)}%</span>
                      </p>
                    </div>

                    <div className="bg-slate-900/50 border border-slate-700 rounded-2xl p-6">
                      <h4 className="font-bold text-white mb-4 text-lg">📊 Métricas de Detección</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs text-slate-500 mb-1">Grietas Detectadas</p>
                          <p className="text-2xl font-bold text-cyan-400">{result.metricas.num_grietas_detectadas}</p>
                        </div>
                        <div>
                          <p className="text-xs text-slate-500 mb-1">Área Afectada</p>
                          <p className="text-2xl font-bold text-orange-400">{result.metricas.porcentaje_grietas. toFixed(2)}%</p>
                        </div>
                        {result.metricas.longitud_total_px && (
                          <div>
                            <p className="text-xs text-slate-500 mb-1">Longitud Total</p>
                            <p className="text-lg font-bold text-blue-400">{result.metricas.longitud_total_px. toFixed(0)} px</p>
                          </div>
                        )}
                        {result.metricas.ancho_promedio_px && (
                          <div>
                            <p className="text-xs text-slate-500 mb-1">Ancho Promedio</p>
                            <p className="text-lg font-bold text-purple-400">{result.metricas.ancho_promedio_px.toFixed(1)} px</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {result.metricas.analisis_morfologico && (
                      <div className="bg-gradient-to-br from-orange-500/10 to-red-500/10 border border-orange-500/30 rounded-2xl p-6">
                        <h4 className="font-bold text-orange-400 mb-4 text-lg flex items-center gap-2">
                          {getPatronIcon(result.metricas.analisis_morfologico. patron_general)}
                          Análisis Morfológico
                        </h4>
                        
                        <div className="space-y-4">
                          <div>
                            <p className="text-sm text-slate-400 mb-1">Patrón Detectado</p>
                            <p className="text-lg font-bold text-white">{result.metricas.analisis_morfologico.patron_general. replace('_', ' ').toUpperCase()}</p>
                            <p className="text-sm text-slate-300 mt-2">{result.metricas.analisis_morfologico.descripcion_patron}</p>
                          </div>

                          <div className="bg-slate-900/50 border border-orange-500/20 rounded-xl p-4">
                            <p className="text-sm text-orange-400 font-semibold mb-2">⚠️ Causa Probable</p>
                            <p className="text-sm text-slate-300">{result. metricas.analisis_morfologico.causa_probable}</p>
                          </div>

                          <div className="bg-slate-900/50 border border-cyan-500/20 rounded-xl p-4">
                            <p className="text-sm text-cyan-400 font-semibold mb-2">💡 Recomendación</p>
                            <p className="text-sm text-slate-300">{result.metricas.analisis_morfologico.recomendacion}</p>
                          </div>

                          <div>
                            <p className="text-sm text-slate-400 mb-3">Distribución de Orientaciones</p>
                            <div className="grid grid-cols-2 gap-2">
                              {Object.entries(result.metricas.analisis_morfologico. distribucion_orientaciones).map(([key, val]) => (
                                <div key={key} className={`${getOrientacionColor(key)} border px-3 py-2 rounded-lg text-center`}>
                                  <p className="text-xs font-semibold uppercase">{key}</p>
                                  <p className="text-lg font-bold">{val}</p>
                                </div>
                              ))}
                            </div>
                          </div>

                          {result.metricas.analisis_morfologico.grietas_principales. length > 0 && (
                            <div>
                              <p className="text-sm text-slate-400 mb-3">Grietas Principales ({result.metricas.analisis_morfologico.grietas_principales.length})</p>
                              <div className="space-y-2 max-h-48 overflow-y-auto">
                                {result.metricas.analisis_morfologico. grietas_principales.slice(0, 5).map((grieta) => (
                                  <div key={grieta.id} className="bg-slate-900/70 border border-slate-700 rounded-lg p-3 text-xs">
                                    <div className="flex items-center justify-between mb-2">
                                      <span className={`${getOrientacionColor(grieta.orientacion)} px-2 py-1 rounded-md font-semibold text-xs border`}>
                                        {grieta.orientacion. toUpperCase()}
                                      </span>
                                      <span className="text-slate-400">Grieta #{grieta.id}</span>
                                    </div>
                                    <div className="grid grid-cols-2 gap-2 text-slate-300">
                                      <div>
                                        <span className="text-slate-500">Longitud:</span> {safeToFixed(grieta.longitud_px, 0)} px
                                      </div>
                                      <div>
                                        <span className="text-slate-500">Ancho:</span> {safeToFixed(grieta.ancho_promedio_px, 1)} px
                                      </div>
                                      {grieta.angulo_grados !== null && (
                                        <div>
                                          <span className="text-slate-500">Ángulo:</span> {safeToFixed(grieta.angulo_grados, 0)}°
                                        </div>
                                      )}
                                      <div>
                                        <span className="text-slate-500">Área:</span> {safeToFixed(grieta.area_px, 0)} px²
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {result.procesamiento && (
                      <div className="bg-slate-900/50 border border-slate-700 rounded-2xl p-6">
                        <h4 className="font-bold text-slate-300 mb-4 text-lg">⚙️ Información de Procesamiento</h4>
                        <div className="space-y-2 text-sm text-slate-400">
                          <p><span className="text-slate-500">Arquitectura:</span> <span className="text-white font-semibold">{result.procesamiento.architecture}</span></p>
                          <p><span className="text-slate-500">Encoder:</span> <span className="text-white font-semibold">{result.procesamiento.encoder}</span></p>
                          <p><span className="text-slate-500">TTA:</span> <span className={result.procesamiento.tta_usado ? 'text-cyan-400' : 'text-slate-400'}>
                            {result.procesamiento.tta_usado ? `✓ Activado (${result.procesamiento.tta_transforms}x)` : '✗ Desactivado'}
                          </span></p>
                          <p><span className="text-slate-500">Threshold:</span> <span className="text-white">{result.procesamiento.threshold}</span></p>
                          {result.procesamiento.cpu_optimized && (
                            <p><span className="text-slate-500">Optimización CPU:</span> <span className="text-green-400">✓ Activado ({result.procesamiento.cpu_threads} threads)</span></p>
                          )}
                          {result.procesamiento.original_dimensions && (
                            <p><span className="text-slate-500">Dimensiones originales:</span> <span className="text-white">
                              {result.procesamiento.original_dimensions.width} × {result.procesamiento.original_dimensions.height} px
                            </span></p>
                          )}
                          {result.procesamiento.output_format && (
                            <p><span className="text-slate-500">Formato salida:</span> <span className="text-white">{result.procesamiento.output_format}</span></p>
                          )}
                        </div>
                      </div>
                    )}

                    <div className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-2xl p-6">
                      <h4 className="font-bold text-green-400 mb-3 text-lg">✅ Análisis Completado</h4>
                      <p className="text-sm text-slate-300 mb-3">
                        El sistema ha detectado y clasificado las grietas en la estructura.   
                        Revisa las recomendaciones y considera una inspección profesional si es necesario.
                      </p>
                      <p className="text-xs text-slate-500">
                        Timestamp: {new Date(result.timestamp).toLocaleString('es-PE')}
                      </p>
                      <p className="text-xs text-green-400 mt-2 flex items-center gap-1">
                        <ZapIcon className="w-3 h-3" />
                        Procesado con Backend v5.0 Multipart (3× más rápido)
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Pruebas

             