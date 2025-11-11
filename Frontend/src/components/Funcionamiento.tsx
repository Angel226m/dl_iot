/*import { Cpu, Server, Monitor, Users, Building2, Shield, Code, Server as ServerIcon, Package, GraduationCap, Camera, Database, Sparkles, ScanSearch, Eye, Calendar, CheckCircle, Clock, ArrowRight, Layers } from 'lucide-react'

const Funcionamiento = () => {
  const layers = [
    {
      icon: <Cpu className="w-10 h-10" />,
      title: 'Capa de Hardware',
      technology: 'Raspberry Pi 4 + Arducam',
      color: 'from-cyan-500 to-blue-600',
      description: 'Dispositivos IoT para captura de imágenes en tiempo real',
      features: ['Raspberry Pi 4', 'Cámara de alta resolución', 'Sensores ambientales'],
    },
    {
      icon: <Server className="w-10 h-10" />,
      title: 'Capa de Servidor',
      technology: 'Python + Flask + TensorFlow',
      color: 'from-blue-500 to-indigo-600',
      description: 'Procesamiento y análisis con modelos de Deep Learning',
      features: ['API Flask', 'Modelo U-Net', 'Optimización con TensorFlow'],
    },
    {
      icon: <Monitor className="w-10 h-10" />,
      title: 'Capa de Cliente',
      technology: 'React + TypeScript + Tailwind',
      color: 'from-indigo-500 to-cyan-600',
      description: 'Interfaz web responsiva para visualización de resultados',
      features: ['React 18', 'TypeScript', 'Diseño adaptativo'],
    },
  ]

  const stakeholders = [
    { icon: <Users className="w-8 h-8" />, title: 'Ingenieros Estructurales', description: 'Análisis y mantenimiento de estructuras', color: 'from-cyan-500 to-blue-500' },
    { icon: <Building2 className="w-8 h-8" />, title: 'Propietarios', description: 'Gestión y monitoreo de edificaciones', color: 'from-green-500 to-emerald-500' },
    { icon: <Shield className="w-8 h-8" />, title: 'Autoridades Municipales', description: 'Regulación y cumplimiento de normas', color: 'from-red-500 to-orange-500' },
    { icon: <Code className="w-8 h-8" />, title: 'Equipo de Desarrollo', description: 'Diseño y desarrollo de CrackGuard', color: 'from-blue-500 to-indigo-500' },
    { icon: <ServerIcon className="w-8 h-8" />, title: 'Administradores de Servidores', description: 'Mantenimiento de infraestructura IT', color: 'from-yellow-500 to-orange-500' },
    { icon: <Package className="w-8 h-8" />, title: 'Proveedores', description: 'Suministro de hardware y software', color: 'from-cyan-500 to-teal-500' },
    { icon: <GraduationCap className="w-8 h-8" />, title: 'Universidad Nacional de Cañete', description: 'Supervisión académica y evaluación', color: 'from-indigo-500 to-blue-600' },
  ]

  const flowSteps = [
    { icon: <Camera className="w-10 h-10" />, title: 'Captura de Datos', description: 'Raspberry Pi con Arducam captura imágenes de estructuras', number: 1, color: 'from-cyan-500 to-blue-500' },
    { icon: <Database className="w-10 h-10" />, title: 'Entrenamiento del Modelo', description: 'Modelo U-Net entrenado con dataset de grietas', number: 2, color: 'from-green-500 to-emerald-500' },
    { icon: <Sparkles className="w-10 h-10" />, title: 'Extracción de Características', description: 'Red neuronal identifica patrones relevantes', number: 3, color: 'from-blue-500 to-indigo-500' },
    { icon: <ScanSearch className="w-10 h-10" />, title: 'Detección y Segmentación', description: 'Algoritmo U-Net delimita áreas con grietas', number: 4, color: 'from-orange-500 to-red-500' },
    { icon: <Eye className="w-10 h-10" />, title: 'Visualización de Resultados', description: 'Interfaz web muestra resultados con métricas', number: 5, color: 'from-indigo-500 to-cyan-500' },
  ]

  const timeline = [
    { week: 'Semana 1-2', dates: '06 Oct - 19 Oct 2025', activities: ['Investigación y planificación', 'Definición de requisitos', 'Diseño de arquitectura'], responsible: 'Todo el equipo', deliverable: 'Documento de especificaciones' },
    { week: 'Semana 3-4', dates: '20 Oct - 02 Nov 2025', activities: ['Configuración de Raspberry Pi', 'Desarrollo backend Flask', 'Recolección de dataset'], responsible: 'Garay, Figueroa', deliverable: 'Prototipo de hardware y API' },
    { week: 'Semana 5', dates: '03 Nov - 09 Nov 2025', activities: ['Entrenamiento del modelo U-Net', 'Optimización de hiperparámetros', 'Validación del modelo'], responsible: 'Borjas, Conca', deliverable: 'Modelo entrenado' },
    { week: 'Semana 6', dates: '10 Nov - 16 Nov 2025', activities: ['Desarrollo del frontend React', 'Integración con backend', 'Diseño de interfaz'], responsible: 'Figueroa, Garay', deliverable: 'Aplicación web funcional' },
    { week: 'Semana 7', dates: '17 Nov - 23 Nov 2025', activities: ['Pruebas integrales', 'Corrección de bugs', 'Documentación final', 'Preparación de presentación'], responsible: 'Todo el equipo', deliverable: 'Sistema completo y documentación' },
  ]

  return (
    <div className="pt-16 bg-slate-950">
      {/* Arquitectura de Tres Capas *//*}
      <section className="relative bg-slate-900 py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full px-5 py-2 mb-6">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">ARQUITECTURA</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Arquitectura de Tres Capas</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Sistema modular con integración escalable y eficiente
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            {layers.map((layer, index) => (
              <div key={index} className="group relative">
                <div className={`absolute inset-0 bg-gradient-to-br ${layer.color} rounded-2xl blur-xl opacity-0 group-hover:opacity-50 transition duration-500`}></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300 h-full">
                  <div className={`bg-gradient-to-br ${layer.color} text-white p-4 rounded-xl mb-4 inline-flex shadow-lg`}>
                    {layer.icon}
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2">{layer.title}</h3>
                  <p className="text-sm font-semibold text-cyan-400 mb-3">{layer.technology}</p>
                  <p className="text-slate-400 mb-4 leading-relaxed">{layer.description}</p>
                  <div className="space-y-2">
                    {layer.features.map((feature, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-sm text-slate-300">
                        <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full"></div>
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Diagrama de flujo *//*}
          <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8 hover:border-cyan-500/50 transition-all duration-300">
            <h3 className="text-2xl font-bold text-center mb-8 text-white">Flujo de Comunicación</h3>
            <div className="flex flex-col md:flex-row items-center justify-center gap-6 md:gap-8">
              <div className="flex flex-col items-center group">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full blur-xl opacity-75 group-hover:opacity-100 transition duration-300"></div>
                  <div className="relative w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-cyan-500/50">
                    1
                  </div>
                </div>
                <p className="mt-3 font-semibold text-white">Hardware</p>
              </div>
              <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
              <div className="text-4xl text-slate-600 md:hidden">↓</div>
              <div className="flex flex-col items-center group">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full blur-xl opacity-75 group-hover:opacity-100 transition duration-300"></div>
                  <div className="relative w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-blue-500/50">
                    2
                  </div>
                </div>
                <p className="mt-3 font-semibold text-white">Servidor</p>
              </div>
              <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
              <div className="text-4xl text-slate-600 md:hidden">↓</div>
              <div className="flex flex-col items-center group">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-cyan-600 rounded-full blur-xl opacity-75 group-hover:opacity-100 transition duration-300"></div>
                  <div className="relative w-20 h-20 bg-gradient-to-br from-indigo-500 to-cyan-600 rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-indigo-500/50">
                    3
                  </div>
                </div>
                <p className="mt-3 font-semibold text-white">Cliente</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stakeholders *//*}
      <section className="relative bg-slate-950 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/30 rounded-full px-5 py-2 mb-6">
              <Users className="w-4 h-4 text-blue-400" />
              <span className="text-blue-400 text-sm font-semibold tracking-wide">STAKEHOLDERS</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Actores y Stakeholders</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Partes interesadas clave en el desarrollo de CrackGuard
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {stakeholders.map((stakeholder, index) => (
              <div key={index} className="group relative">
                <div className={`absolute inset-0 bg-gradient-to-br ${stakeholder.color} rounded-2xl blur-xl opacity-0 group-hover:opacity-50 transition duration-500`}></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300 h-full transform group-hover:-translate-y-2">
                  <div className={`bg-gradient-to-br ${stakeholder.color} text-white p-3 rounded-xl mb-4 inline-flex shadow-lg`}>
                    {stakeholder.icon}
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{stakeholder.title}</h3>
                  <p className="text-slate-400 text-sm leading-relaxed">{stakeholder.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Flujo de Detección *//*}
      <section className="relative bg-slate-900 py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-green-500/10 border border-green-500/30 rounded-full px-5 py-2 mb-6">
              <Sparkles className="w-4 h-4 text-green-400" />
              <span className="text-green-400 text-sm font-semibold tracking-wide">PROCESO</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Flujo de Detección</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Proceso completo desde la captura hasta la visualización de resultados
            </p>
          </div>

          <div className="space-y-6">
            {flowSteps.map((step, index) => (
              <div key={index} className="group">
                <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
                  <div className="relative flex-shrink-0">
                    <div className={`absolute inset-0 bg-gradient-to-br ${step.color} rounded-full blur-xl opacity-75 group-hover:opacity-100 transition duration-300`}></div>
                    <div className={`relative w-20 h-20 bg-gradient-to-br ${step.color} rounded-full flex items-center justify-center text-white shadow-lg`}>
                      {step.icon}
                    </div>
                  </div>
                  
                  <div className="flex-grow">
                    <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300">
                      <div className="flex items-center gap-3 mb-2">
                        <span className={`text-3xl font-black bg-gradient-to-br ${step.color} bg-clip-text text-transparent`}>
                          {step.number}
                        </span>
                        <h3 className="text-xl md:text-2xl font-bold text-white">{step.title}</h3>
                      </div>
                      <p className="text-slate-400 text-base md:text-lg leading-relaxed">{step.description}</p>
                    </div>
                  </div>
                </div>

                {index < flowSteps.length - 1 && (
                  <div className="flex justify-center my-4">
                    <div className="text-4xl text-slate-700">↓</div>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="mt-16 relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-600/20 rounded-3xl blur-2xl opacity-75 group-hover:opacity-100 transition duration-500"></div>
            <div className="relative bg-gradient-to-br from-slate-800 via-slate-900 to-slate-950 border border-slate-700 rounded-3xl p-8 md:p-12">
              <h3 className="text-3xl font-black text-white text-center mb-8">Deep Learning en Acción</h3>
              <div className="flex flex-col md:flex-row justify-around items-center gap-8 md:gap-6">
                <div className="text-center group/item">
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl mb-3 hover:border-cyan-500/50 transition-all duration-300">
                    <Database className="w-16 h-16 text-cyan-400 mx-auto" />
                  </div>
                  <p className="font-bold text-white text-lg">Entrada</p>
                  <p className="text-sm text-slate-400">Imágenes</p>
                </div>
                <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
                <div className="text-4xl text-slate-600 md:hidden">↓</div>
                <div className="text-center group/item">
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl mb-3 hover:border-blue-500/50 transition-all duration-300">
                    <Cpu className="w-16 h-16 text-blue-400 mx-auto" />
                  </div>
                  <p className="font-bold text-white text-lg">Procesamiento</p>
                  <p className="text-sm text-slate-400">U-Net Model</p>
                </div>
                <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
                <div className="text-4xl text-slate-600 md:hidden">↓</div>
                <div className="text-center group/item">
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl mb-3 hover:border-indigo-500/50 transition-all duration-300">
                    <Eye className="w-16 h-16 text-indigo-400 mx-auto" />
                  </div>
                  <p className="font-bold text-white text-lg">Salida</p>
                  <p className="text-sm text-slate-400">Detección</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Cronograma *//*}
      <section className="relative bg-slate-950 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/30 rounded-full px-5 py-2 mb-6">
              <Calendar className="w-4 h-4 text-blue-400" />
              <span className="text-blue-400 text-sm font-semibold tracking-wide">PLANIFICACIÓN</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Cronograma del Proyecto</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Planificación de 7 semanas (06 Oct - 23 Nov 2025)
            </p>
          </div>

          <div className="space-y-6">
            {timeline.map((item, index) => (
              <div key={index} className="group relative">
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300">
                  <div className="flex flex-col lg:flex-row gap-6">
                    <div className="flex-shrink-0">
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full blur-xl opacity-75"></div>
                        <div className="relative w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center text-white font-black text-2xl shadow-lg shadow-cyan-500/50">
                          {index + 1}
                        </div>
                      </div>
                    </div>

                    <div className="flex-grow space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                        <h3 className="text-2xl font-bold text-white">{item.week}</h3>
                        <div className="flex items-center gap-2 text-slate-400">
                          <Calendar className="w-5 h-5 text-cyan-400" />
                          <span className="font-medium">{item.dates}</span>
                        </div>
                      </div>

                      <div className="grid sm:grid-cols-2 gap-4">
                        <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-4">
                          <h4 className="font-semibold text-white mb-3 flex items-center gap-2">
                            <Clock className="w-4 h-4 text-blue-400" />
                            Actividades:
                          </h4>
                          <ul className="space-y-2">
                            {item.activities.map((activity, idx) => (
                              <li key={idx} className="flex items-start gap-2 text-slate-300 text-sm">
                                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                                <span>{activity}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div className="space-y-4">
                          <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-4">
                            <h4 className="font-semibold text-white mb-2 text-sm">👥 Responsables:</h4>
                            <p className="text-cyan-400 font-medium">{item.responsible}</p>
                          </div>
                          
                          <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
                            <h4 className="font-semibold text-white mb-2 text-sm">📦 Entregable:</h4>
                            <p className="text-blue-400 font-medium">{item.deliverable}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

export default Funcionamiento*/

 import { Cpu, Server, Monitor, Users, Building2, Shield, Code, Server as ServerIcon, Package, GraduationCap, Camera, Database, Sparkles, ScanSearch, Eye, Calendar, CheckCircle, Clock, ArrowRight, Layers, Compass, Settings, Zap } from 'lucide-react'

const Funcionamiento = () => {
  const layers = [
    {
      icon: <Camera className="w-12 h-12" />,
      title: 'Capa de Hardware',
      subtitle: 'Captura y Adquisición de Datos',
      technology: 'Raspberry Pi 4 + Arducam',
      color: 'from-cyan-500 to-blue-600',
      description: 'Infraestructura IoT para captura de imágenes en alta resolución en tiempo real, con simulación de entornos para correlación de datos externos. Optimiza la calidad visual para análisis preciso de grietas mediante procesamiento contextual y enriquecimiento de datos.',
      features: [
        'Raspberry Pi 4 (8GB RAM, Quad-Core Cortex-A72)',
        'Cámara Arducam 1080p con soporte HDR',
        'Conexión WiFi para transmisión de datos',
        'Simulación de captura para pruebas en entornos controlados',
        'Soporte para formatos PNG, JPG, BMP, TIFF (máx. 20MB)',
      ],
      metrics: [
        { label: 'Resolución', value: '1920x1080 px', icon: '📷' },
        { label: 'Frecuencia', value: '30 FPS', icon: '⚡' },
        { label: 'Tamaño Máximo', value: '20 MB', icon: '💾' },
      ],
    },
    {
      icon: <Server className="w-12 h-12" />,
      title: 'Capa de Servidor',
      subtitle: 'Procesamiento y Análisis IA',
      technology: 'Python + Flask + TensorFlow',
      color: 'from-blue-500 to-indigo-600',
      description: 'Backend robusto para procesamiento de imágenes con modelos de deep learning avanzados. Implementa UNet++ con EfficientNet-B8, Test-Time Augmentation (TTA) y análisis morfológico para una detección precisa de grietas.',
      features: [
        'API REST con1 Flask para predicciones en tiempo real',
        'Modelo UNet++ con backbone EfficientNet-B8',
        'TTA con 6 transformaciones para mayor precisión',
        'Análisis morfológico avanzado (patrones, orientaciones, severidad)',
        'Optimización con TensorFlow para alto rendimiento',
        'Soporte para imágenes de hasta 2048x2048 px',
      ],
      metrics: [
        { label: 'TTA Transformaciones', value: '6x', icon: '🔄' },
        { label: 'Umbral de Detección', value: '0.5', icon: '🎯' },
        { label: 'Tiempo de Procesamiento', value: '~2s por imagen', icon: '⏱️' },
      ],
    },
    {
      icon: <Monitor className="w-12 h-12" />,
      title: 'Capa de Cliente',
      subtitle: 'Interfaz y Visualización',
      technology: 'React + TypeScript + Tailwind',
      color: 'from-indigo-500 to-cyan-600',
      description: 'Interfaz web responsiva que permite cargar imágenes, capturar fotos en tiempo real, y visualizar resultados con métricas detalladas y overlays de grietas. Diseñada para ser intuitiva y accesible en múltiples dispositivos.',
      features: [
        'React 18 con hooks para una experiencia fluida',
        'TypeScript para tipado seguro y escalable',
        'Diseño responsivo con Tailwind CSS',
        'Integración con cámara web y dispositivos móviles',
        'Visualización interactiva de métricas y análisis morfológico',
        'Soporte para imágenes procesadas con overlay de grietas',
      ],
      metrics: [
        { label: 'Compatibilidad', value: 'Web/Móvil', icon: '📱' },
        { label: 'Tiempo de Render', value: '<100ms', icon: '⚡' },
        { label: 'Formatos Soportados', value: '5+', icon: '🖼️' },
      ],
    },
  ]

  const stakeholders = [
    { icon: <Users className="w-8 h-8" />, title: 'Ingenieros Estructurales', description: 'Análisis detallado de grietas con métricas morfológicas y severidad', color: 'from-cyan-500 to-blue-500' },
    { icon: <Building2 className="w-8 h-8" />, title: 'Propietarios', description: 'Monitoreo en tiempo real con alertas de severidad', color: 'from-green-500 to-emerald-500' },
    { icon: <Shield className="w-8 h-8" />, title: 'Autoridades Municipales', description: 'Reportes de cumplimiento basados en análisis IA', color: 'from-red-500 to-orange-500' },
    { icon: <Code className="w-8 h-8" />, title: 'Equipo de Desarrollo', description: 'Diseño y optimización de CrackGuard con TTA', color: 'from-blue-500 to-indigo-500' },
    { icon: <ServerIcon className="w-8 h-8" />, title: 'Administradores de Servidores', description: 'Gestión de API y modelos de IA escalables', color: 'from-yellow-500 to-orange-500' },
    { icon: <Package className="w-8 h-8" />, title: 'Proveedores', description: 'Suministro de hardware IoT y librerías de ML', color: 'from-cyan-500 to-teal-500' },
    { icon: <GraduationCap className="w-8 h-8" />, title: 'Universidad Nacional de Cañete', description: 'Supervisión académica y validación de modelos', color: 'from-indigo-500 to-blue-600' },
  ]

  const flowSteps = [
    { icon: <Camera className="w-10 h-10" />, title: 'Captura de Datos', description: 'Raspberry Pi con Arducam captura imágenes de estructuras, soporta formatos PNG, JPG, BMP, TIFF hasta 20MB', number: 1, color: 'from-cyan-500 to-blue-500' },
    { icon: <Database className="w-10 h-10" />, title: 'Entrenamiento del Modelo', description: 'Modelo UNet++ EfficientNet-B8 entrenado con dataset de grietas', number: 2, color: 'from-green-500 to-emerald-500' },
    { icon: <Sparkles className="w-10 h-10" />, title: 'Extracción de Características', description: 'Red neuronal identifica patrones con TTA (6x) para mayor precisión', number: 3, color: 'from-blue-500 to-indigo-500' },
    { icon: <ScanSearch className="w-10 h-10" />, title: 'Detección y Segmentación', description: 'Algoritmo UNet++ delimita áreas con grietas, calcula métricas como longitud y ancho', number: 4, color: 'from-orange-500 to-red-500' },
    { icon: <Compass className="w-10 h-10" />, title: 'Análisis Morfológico', description: 'Evaluación de patrones (horizontal, vertical, diagonal), causas probables y severidad ajustada', number: 5, color: 'from-purple-500 to-pink-500' },
    { icon: <Eye className="w-10 h-10" />, title: 'Visualización de Resultados', description: 'Interfaz muestra overlay de grietas, métricas detalladas y recomendaciones', number: 6, color: 'from-indigo-500 to-cyan-500' },
  ]

  const timeline = [
    { week: 'Semana 1-2', dates: '06 Oct - 19 Oct 2025', activities: ['Investigación y planificación', 'Definición de requisitos', 'Diseño de arquitectura'], responsible: 'Todo el equipo', deliverable: 'Documento de especificaciones', status: 'Completado' },
    { week: 'Semana 3-4', dates: '20 Oct - 02 Nov 2025', activities: ['Configuración de Raspberry Pi', 'Desarrollo backend Flask', 'Recolección de dataset', 'Integración de TTA y análisis morfológico'], responsible: 'Garay, Figueroa', deliverable: 'Prototipo de hardware y API', status: 'En Progreso' },
    { week: 'Semana 5', dates: '03 Nov - 09 Nov 2025', activities: ['Entrenamiento del modelo UNet++ EfficientNet-B8', 'Optimización de hiperparámetros', 'Validación del modelo con TTA'], responsible: 'Borjas, Conca', deliverable: 'Modelo entrenado', status: 'Pendiente' },
    { week: 'Semana 6', dates: '10 Nov - 16 Nov 2025', activities: ['Desarrollo del frontend React', 'Integración con backend', 'Diseño de interfaz para pruebas en vivo'], responsible: 'Figueroa, Garay', deliverable: 'Aplicación web funcional', status: 'Pendiente' },
    { week: 'Semana 7', dates: '17 Nov - 23 Nov 2025', activities: ['Pruebas integrales con imágenes reales', 'Corrección de bugs', 'Documentación final', 'Preparación de presentación'], responsible: 'Todo el equipo', deliverable: 'Sistema completo y documentación', status: 'Pendiente' },
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Completado': return 'bg-green-500/10 border-green-500/30 text-green-400'
      case 'En Progreso': return 'bg-blue-500/10 border-blue-500/30 text-blue-400'
      case 'Pendiente': return 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400'
      default: return 'bg-slate-500/10 border-slate-500/30 text-slate-400'
    }
  }

  return (
    <div className="pt-16 bg-slate-950">
      {/* Arquitectura de Tres Capas */}
      <section className="relative bg-slate-900 py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full px-5 py-2 mb-6">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">ARQUITECTURA</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">
              Arquitectura de Tres Capas
            </h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Sistema modular y escalable que combina hardware IoT, procesamiento de IA con TTA y análisis morfológico, y una interfaz web interactiva para detección de grietas en tiempo real.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
            {layers.map((layer, index) => (
              <div key={index} className="group relative">
                <div className={`absolute inset-0 bg-gradient-to-br ${layer.color} rounded-3xl blur-2xl opacity-0 group-hover:opacity-50 transition duration-500`}></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-8 hover:border-cyan-500/50 transition-all duration-300 transform group-hover:-translate-y-2">
                  <div className={`bg-gradient-to-br ${layer.color} text-white p-4 rounded-xl mb-6 inline-flex shadow-lg shadow-cyan-500/50`}>
                    {layer.icon}
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-2">{layer.title}</h3>
                  <p className="text-sm font-semibold text-cyan-400 mb-3">{layer.subtitle}</p>
                  <p className="text-base text-slate-300 mb-6 leading-relaxed">{layer.description}</p>
                  
                  <div className="mb-6">
                    <h4 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                      <Settings className="w-5 h-5 text-cyan-400" />
                      Características
                    </h4>
                    <div className="space-y-3">
                      {layer.features.map((feature, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>{feature}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-5">
                    <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Zap className="w-5 h-5 text-purple-400" />
                      Métricas Clave
                    </h4>
                    <div className="grid grid-cols-2 gap-3">
                      {layer.metrics.map((metric, idx) => (
                        <div key={idx} className="bg-slate-800/50 border border-slate-600 rounded-lg p-3 text-center">
                          <p className="text-xs text-slate-400 mb-1 flex items-center justify-center gap-1">
                            <span>{metric.icon}</span>
                            {metric.label}
                          </p>
                          <p className="text-sm font-bold text-white">{metric.value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-600/20 rounded-3xl blur-2xl opacity-75 group-hover:opacity-100 transition duration-500"></div>
            <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-8 md:p-12 hover:border-cyan-500/50 transition-all duration-300">
              <h3 className="text-3xl font-black text-white text-center mb-8">
                Flujo de Comunicación
              </h3>
              <p className="text-center text-slate-400 mb-8 max-w-3xl mx-auto">
                Interacción fluida entre capas para un procesamiento eficiente: desde la captura de imágenes hasta la visualización de resultados con análisis morfológico.
              </p>
              <div className="flex flex-col md:flex-row items-center justify-center gap-6 md:gap-8">
                <div className="flex flex-col items-center group/item">
                  <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full blur-xl opacity-75 group-hover/item:opacity-100 transition duration-300"></div>
                    <div className="relative w-24 h-24 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center text-white shadow-lg shadow-cyan-500/50">
                      <Camera className="w-10 h-10" />
                    </div>
                  </div>
                  <p className="mt-3 font-semibold text-white">Captura (Hardware)</p>
                  <p className="text-sm text-slate-400 text-center">Imágenes en alta resolución</p>
                </div>
                <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
                <div className="text-4xl text-slate-600 md:hidden">↓</div>
                <div className="flex flex-col items-center group/item">
                  <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full blur-xl opacity-75 group-hover/item:opacity-100 transition duration-300"></div>
                    <div className="relative w-24 h-24 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center text-white shadow-lg shadow-blue-500/50">
                      <Server className="w-10 h-10" />
                    </div>
                  </div>
                  <p className="mt-3 font-semibold text-white">Procesamiento (Servidor)</p>
                  <p className="text-sm text-slate-400 text-center">UNet++ con TTA y análisis morfológico</p>
                </div>
                <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
                <div className="text-4xl text-slate-600 md:hidden">↓</div>
                <div className="flex flex-col items-center group/item">
                  <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-cyan-600 rounded-full blur-xl opacity-75 group-hover/item:opacity-100 transition duration-300"></div>
                    <div className="relative w-24 h-24 bg-gradient-to-br from-indigo-500 to-cyan-600 rounded-full flex items-center justify-center text-white shadow-lg shadow-indigo-500/50">
                      <Monitor className="w-10 h-10" />
                    </div>
                  </div>
                  <p className="mt-3 font-semibold text-white">Visualización (Cliente)</p>
                  <p className="text-sm text-slate-400 text-center">Interfaz con métricas y overlays</p>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-12 relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-600/20 rounded-3xl blur-2xl opacity-75 group-hover:opacity-100 transition duration-500"></div>
            <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-3xl p-8 md:p-12 hover:border-purple-500/50 transition-all duration-300">
              <h3 className="text-3xl font-black text-white text-center mb-8">
                Detalles Técnicos
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-6">
                  <h4 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <Compass className="w-6 h-6 text-purple-400" />
                    Análisis Morfológico
                  </h4>
                  <p className="text-slate-300 mb-4">
                    Identifica patrones de grietas (horizontal, vertical, diagonal, ramificada) y calcula métricas como longitud, ancho, ángulo y severidad ajustada.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {['Horizontal ↔️', 'Vertical ↕️', 'Diagonal ↗️', 'Ramificada 🗺️'].map((pattern, idx) => (
                      <span key={idx} className="bg-purple-500/10 border border-purple-500/30 text-purple-400 text-xs font-semibold px-3 py-1 rounded-full">
                        {pattern}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-6">
                  <h4 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <Zap className="w-6 h-6 text-cyan-400" />
                    Test-Time Augmentation (TTA)
                  </h4>
                  <p className="text-slate-300 mb-4">
                    Aplica 6 transformaciones (rotaciones, flips, escalado) para mejorar la robustez del modelo, aumentando la confianza en la detección hasta un 95%+.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {['Rotaciones', 'Flips', 'Escalado', '6x Aumentos'].map((aug, idx) => (
                      <span key={idx} className="bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold px-3 py-1 rounded-full">
                        {aug}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stakeholders */}
      <section className="relative bg-slate-950 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/30 rounded-full px-5 py-2 mb-6">
              <Users className="w-4 h-4 text-blue-400" />
              <span className="text-blue-400 text-sm font-semibold tracking-wide">STAKEHOLDERS</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Actores y Stakeholders</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Partes interesadas clave en el desarrollo y uso de CrackGuard
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {stakeholders.map((stakeholder, index) => (
              <div key={index} className="group relative">
                <div className={`absolute inset-0 bg-gradient-to-br ${stakeholder.color} rounded-2xl blur-xl opacity-0 group-hover:opacity-50 transition duration-500`}></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300 h-full transform group-hover:-translate-y-2">
                  <div className={`bg-gradient-to-br ${stakeholder.color} text-white p-3 rounded-xl mb-4 inline-flex shadow-lg`}>
                    {stakeholder.icon}
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{stakeholder.title}</h3>
                  <p className="text-slate-400 text-sm leading-relaxed">{stakeholder.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Flujo de Detección */}
      <section className="relative bg-slate-900 py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-green-500/10 border border-green-500/30 rounded-full px-5 py-2 mb-6">
              <Sparkles className="w-4 h-4 text-green-400" />
              <span className="text-green-400 text-sm font-semibold tracking-wide">PROCESO</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Flujo de Detección</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Proceso completo desde la captura hasta el análisis morfológico y visualización
            </p>
          </div>

          <div className="space-y-6">
            {flowSteps.map((step, index) => (
              <div key={index} className="group">
                <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
                  <div className="relative flex-shrink-0">
                    <div className={`absolute inset-0 bg-gradient-to-br ${step.color} rounded-full blur-xl opacity-75 group-hover:opacity-100 transition duration-300`}></div>
                    <div className={`relative w-20 h-20 bg-gradient-to-br ${step.color} rounded-full flex items-center justify-center text-white shadow-lg`}>
                      {step.icon}
                    </div>
                  </div>
                  
                  <div className="flex-grow">
                    <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300">
                      <div className="flex items-center gap-3 mb-2">
                        <span className={`text-3xl font-black bg-gradient-to-br ${step.color} bg-clip-text text-transparent`}>
                          {step.number}
                        </span>
                        <h3 className="text-xl md:text-2xl font-bold text-white">{step.title}</h3>
                      </div>
                      <p className="text-slate-400 text-base md:text-lg leading-relaxed">{step.description}</p>
                    </div>
                  </div>
                </div>

                {index < flowSteps.length - 1 && (
                  <div className="flex justify-center my-4">
                    <div className="text-4xl text-slate-700">↓</div>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="mt-16 relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-600/20 rounded-3xl blur-2xl opacity-75 group-hover:opacity-100 transition duration-500"></div>
            <div className="relative bg-gradient-to-br from-slate-800 via-slate-900 to-slate-950 border border-slate-700 rounded-3xl p-8 md:p-12">
              <h3 className="text-3xl font-black text-white text-center mb-8">Deep Learning en Acción</h3>
              <div className="flex flex-col md:flex-row justify-around items-center gap-8 md:gap-6">
                <div className="text-center group/item">
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl mb-3 hover:border-cyan-500/50 transition-all duration-300">
                    <Database className="w-16 h-16 text-cyan-400 mx-auto" />
                  </div>
                  <p className="font-bold text-white text-lg">Entrada</p>
                  <p className="text-sm text-slate-400">Imágenes</p>
                </div>
                <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
                <div className="text-4xl text-slate-600 md:hidden">↓</div>
                <div className="text-center group/item">
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl mb-3 hover:border-blue-500/50 transition-all duration-300">
                    <Cpu className="w-16 h-16 text-blue-400 mx-auto" />
                  </div>
                  <p className="font-bold text-white text-lg">Procesamiento</p>
                  <p className="text-sm text-slate-400">UNet++ EfficientNet-B8 + TTA</p>
                </div>
                <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
                <div className="text-4xl text-slate-600 md:hidden">↓</div>
                <div className="text-center group/item">
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl mb-3 hover:border-purple-500/50 transition-all duration-300">
                    <Compass className="w-16 h-16 text-purple-400 mx-auto" />
                  </div>
                  <p className="font-bold text-white text-lg">Análisis</p>
                  <p className="text-sm text-slate-400">Morfológico</p>
                </div>
                <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
                <div className="text-4xl text-slate-600 md:hidden">↓</div>
                <div className="text-center group/item">
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl mb-3 hover:border-indigo-500/50 transition-all duration-300">
                    <Eye className="w-16 h-16 text-indigo-400 mx-auto" />
                  </div>
                  <p className="font-bold text-white text-lg">Salida</p>
                  <p className="text-sm text-slate-400">Detección + Métricas</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Cronograma */}
      <section className="relative bg-slate-950 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/30 rounded-full px-5 py-2 mb-6">
              <Calendar className="w-4 h-4 text-blue-400" />
              <span className="text-blue-400 text-sm font-semibold tracking-wide">PLANIFICACIÓN</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Cronograma del Proyecto</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Planificación de 7 semanas (06 Oct - 23 Nov 2025) - Actualizado al 27 Oct 2025
            </p>
          </div>

          <div className="space-y-6">
            {timeline.map((item, index) => (
              <div key={index} className="group relative">
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300">
                  <div className="flex flex-col lg:flex-row gap-6">
                    <div className="flex-shrink-0">
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full blur-xl opacity-75"></div>
                        <div className="relative w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center text-white font-black text-2xl shadow-lg shadow-cyan-500/50">
                          {index + 1}
                        </div>
                      </div>
                    </div>

                    <div className="flex-grow space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                        <h3 className="text-2xl font-bold text-white">{item.week}</h3>
                        <div className="flex items-center gap-2 text-slate-400">
                          <Calendar className="w-5 h-5 text-cyan-400" />
                          <span className="font-medium">{item.dates}</span>
                        </div>
                      </div>

                      <div className="grid sm:grid-cols-2 gap-4">
                        <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-4">
                          <h4 className="font-semibold text-white mb-3 flex items-center gap-2">
                            <Clock className="w-4 h-4 text-blue-400" />
                            Actividades:
                          </h4>
                          <ul className="space-y-2">
                            {item.activities.map((activity, idx) => (
                              <li key={idx} className="flex items-start gap-2 text-slate-300 text-sm">
                                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                                <span>{activity}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div className="space-y-4">
                          <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-4">
                            <h4 className="font-semibold text-white mb-2 text-sm">👥 Responsables:</h4>
                            <p className="text-cyan-400 font-medium">{item.responsible}</p>
                          </div>
                          
                          <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
                            <h4 className="font-semibold text-white mb-2 text-sm">📦 Entregable:</h4>
                            <p className="text-blue-400 font-medium">{item.deliverable}</p>
                          </div>

                          <div className={`rounded-xl p-4 ${getStatusColor(item.status)}`}>
                            <h4 className="font-semibold text-white mb-2 text-sm">📊 Estado:</h4>
                            <p className="font-medium">{item.status}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

export default Funcionamiento