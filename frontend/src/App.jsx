import { useEffect, useState } from 'react'
import {
  Box,
  Flex,
  Heading,
  Text,
  Progress,
  FormControl,
  FormLabel,
  Input,
  Textarea,
  Select,
  Button,
  Image,
  Stack
} from '@chakra-ui/react'

const genresList = [
  'Ação e Aventura',
  'Cartas e Tabuleiro',
  'Clássicos',
  'Família e Crianças',
  'Luta',
  'Indie',
  'Multijogador',
  'Plataformas',
  'Quebra-cabeça e Trivia',
  'Corrida e Voo',
  'RPG',
  'Tiro',
  'Simulação',
  'Esportes',
  'Estratégia',
  'Horror de Sobrevivência',
  'Mundo Aberto',
  'Outros'
]

const modesList = [
  'Single-player',
  'Multiplayer local',
  'Multiplayer online',
  'Cooperativo (Co-op)',
  'Competitivo (PvP)'
]

export default function App() {
  const [index, setIndex] = useState(0)
  const [seq, setSeq] = useState(1)
  const [total, setTotal] = useState(0)
  const [fields, setFields] = useState({
    Name: '',
    Summary: '',
    FirstLaunchDate: '',
    Developers: '',
    Publishers: '',
    Genres: [],
    GameModes: []
  })
  const [image, setImage] = useState('')
  const [uploadName, setUploadName] = useState(null)
  const [done, setDone] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    loadGame()
  }, [])

  useEffect(() => {
    saveSession()
  }, [fields, image, uploadName, index])

  function saveSession() {
    const data = {
      index,
      fields,
      image,
      upload_name: uploadName
    }
    localStorage.setItem('session', JSON.stringify(data))
  }

  function restoreSession(idx) {
    const s = localStorage.getItem('session')
    if (!s) return
    const data = JSON.parse(s)
    if (data.index !== idx) return
    setFields(data.fields)
    if (data.image) setImage(data.image)
    setUploadName(data.upload_name)
  }

  function loadGame() {
    fetch('/api/game')
      .then(r => r.json())
      .then(data => {
        if (data.done) {
          setDone(true)
          setMessage(data.message)
          return
        }
        setIndex(data.index)
        setSeq(data.seq || 1)
        setTotal(data.total || 0)
        setFields({
          Name: data.game.Name || '',
          Summary: data.game.Summary || '',
          FirstLaunchDate: data.game.FirstLaunchDate || '',
          Developers: data.game.Developers || '',
          Publishers: data.game.Publishers || '',
          Genres: Array.isArray(data.game.Genres) ? data.game.Genres : [],
          GameModes: Array.isArray(data.game.GameModes) ? data.game.GameModes : []
        })
        if (data.cover) {
          setImage(data.cover)
        } else {
          setImage('')
        }
        setUploadName(null)
        restoreSession(data.index)
      })
      .catch(err => {
        console.error(err)
        alert('Failed to load game: ' + err.message)
      })
  }

  function handleChange(e) {
    const { name, value, options } = e.target
    if (name === 'Genres' || name === 'GameModes') {
      const vals = Array.from(options).filter(o => o.selected).map(o => o.value)
      setFields(prev => ({ ...prev, [name]: vals }))
    } else {
      setFields(prev => ({ ...prev, [name]: value }))
    }
  }

  function handleUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)
    fetch('/api/upload', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(res => {
        setUploadName(res.filename)
        setImage(res.data)
      })
      .catch(err => {
        console.error(err)
        alert('Failed to upload image: ' + err.message)
      })
  }

  function generateSummary() {
    const btn = document.getElementById('generate-summary')
    if (btn) {
      btn.disabled = true
      btn.textContent = 'Gerando...'
    }
    fetch('/api/summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_name: fields.Name })
    })
      .then(r => r.json())
      .then(res => {
        if (res.summary) {
          setFields(prev => ({ ...prev, Summary: res.summary }))
        } else if (res.error) {
          alert(res.error)
        } else {
          alert('Não foi possível gerar o resumo.')
        }
      })
      .catch(err => {
        console.error(err)
        alert('Erro ao gerar resumo.')
      })
      .finally(() => {
        if (btn) {
          btn.disabled = false
          btn.textContent = 'Gerar Resumo'
        }
      })
  }

  function saveGame() {
    const dataUrl = image
    const payload = {
      index,
      fields,
      image: dataUrl,
      upload_name: uploadName
    }
    fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(r => r.json())
      .then(() => {
        localStorage.removeItem('session')
        setUploadName(null)
        alert('The game was saved.')
      })
      .catch(err => {
        console.error(err)
        alert('Failed to save game: ' + err.message)
      })
  }

  function skipGame() {
    fetch('/api/skip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index, upload_name: uploadName })
    })
      .then(r => r.json())
      .then(() => {
        localStorage.removeItem('session')
        loadGame()
      })
      .catch(err => {
        console.error(err)
        alert('Failed to skip game: ' + err.message)
      })
  }

  function nextGame() {
    fetch('/api/next', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upload_name: uploadName })
    })
      .then(r => r.json())
      .then(() => {
        localStorage.removeItem('session')
        loadGame()
      })
      .catch(err => {
        console.error(err)
        alert('Failed to move to next game: ' + err.message)
      })
  }

  function previousGame() {
    fetch('/api/back', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upload_name: uploadName })
    })
      .then(r => r.json())
      .then(() => {
        localStorage.removeItem('session')
        loadGame()
      })
      .catch(err => {
        console.error(err)
        alert('Failed to move to previous game: ' + err.message)
      })
  }

  function resetFields() {
    fetch(`/api/game/${index}/raw`)
      .then(r => r.json())
      .then(data => {
        setFields({
          Name: data.game.Name || '',
          Summary: data.game.Summary || '',
          FirstLaunchDate: data.game.FirstLaunchDate || '',
          Developers: data.game.Developers || '',
          Publishers: data.game.Publishers || '',
          Genres: Array.isArray(data.game.Genres) ? data.game.Genres : [],
          GameModes: Array.isArray(data.game.GameModes) ? data.game.GameModes : []
        })
        if (data.cover) {
          setImage(data.cover)
        } else {
          setImage('')
        }
        setUploadName(null)
        saveSession()
      })
      .catch(err => {
        console.error(err)
        alert('Failed to reset fields: ' + err.message)
      })
  }

  function revertImage() {
    fetch(`/api/game/${index}/raw`)
      .then(r => r.json())
      .then(data => {
        if (data.cover) {
          setImage(data.cover)
        } else {
          setImage('')
        }
        setUploadName(null)
        saveSession()
      })
      .catch(err => {
        console.error(err)
        alert('Failed to revert image: ' + err.message)
      })
  }

  if (done) {
    return (
      <Box p={4}>
        <Heading size="md">{message}</Heading>
      </Box>
    )
  }

  return (
    <Flex direction="column" p={4} gap={4}>
      <Box>
        <Heading id="game-name">{fields.Name}</Heading>
        <Text id="caption">Processados: {seq - 1} de {total}</Text>
        <Progress value={total ? ((seq - 1) / total) * 100 : 0} />
      </Box>
      <Flex gap={4}>
        <Box flex="1">
          <form id="game-form">
            <Stack spacing={3}>
              <FormControl>
                <Input type="file" onChange={handleUpload} accept="image/*" />
              </FormControl>
              {image && (
                <Image src={image} alt="preview" />
              )}
              <Stack direction="row" spacing={2}>
                <Button onClick={previousGame}>Previous</Button>
                <Button onClick={nextGame}>Next</Button>
                <Button onClick={saveGame} colorScheme="green">Save</Button>
                <Button onClick={skipGame} colorScheme="yellow">Skip</Button>
                <Button onClick={resetFields}>Reset</Button>
                <Button onClick={revertImage}>Revert Image</Button>
              </Stack>
              <FormControl>
                <FormLabel>Name</FormLabel>
                <Input name="Name" value={fields.Name} onChange={handleChange} />
              </FormControl>
              <FormControl>
                <FormLabel>Summary</FormLabel>
                <Textarea name="Summary" value={fields.Summary} onChange={handleChange} />
                <Button id="generate-summary" mt={2} onClick={generateSummary}>Gerar Resumo</Button>
              </FormControl>
              <FormControl>
                <FormLabel>First Launch Date</FormLabel>
                <Input name="FirstLaunchDate" value={fields.FirstLaunchDate} onChange={handleChange} />
              </FormControl>
              <FormControl>
                <FormLabel>Developers</FormLabel>
                <Input name="Developers" value={fields.Developers} onChange={handleChange} />
              </FormControl>
              <FormControl>
                <FormLabel>Publishers</FormLabel>
                <Input name="Publishers" value={fields.Publishers} onChange={handleChange} />
              </FormControl>
              <FormControl>
                <FormLabel>Genres</FormLabel>
                <Select name="Genres" multiple value={fields.Genres} onChange={handleChange} height="auto">
                  {genresList.map(g => <option key={g} value={g}>{g}</option>)}
                </Select>
              </FormControl>
              <FormControl>
                <FormLabel>Game Modes</FormLabel>
                <Select name="GameModes" multiple value={fields.GameModes} onChange={handleChange} height="auto">
                  {modesList.map(m => <option key={m} value={m}>{m}</option>)}
                </Select>
              </FormControl>
            </Stack>
          </form>
        </Box>
      </Flex>
    </Flex>
  )
}
